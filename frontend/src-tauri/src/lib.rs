// SETH — Tauri Desktop App
//
// Spawns the Python WebSocket server as a sidecar process on startup,
// and ensures it is killed when the application exits.

use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandChild;

/// Holds the sidecar child process handle for lifecycle management.
struct SidecarState(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Log plugin only in debug builds
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Spawn the Python server from resources
            // Tauri bundles resources using relative paths, so "../../dist" becomes "_up_/_up_/dist"
            let resource_path_res = app.path().resolve(
                "_up_/_up_/dist/seth-server/seth-server.exe",
                tauri::path::BaseDirectory::Resource
            );
            
            if let Ok(resource_path) = resource_path_res {
                let sidecar_command = app
                    .shell()
                    .command(resource_path.to_string_lossy().to_string());

                match sidecar_command.spawn() {
                    Ok((mut rx, child)) => {
                        // Store child handle so we can kill it on exit
                        app.manage(SidecarState(Mutex::new(Some(child))));

                        // Log sidecar stdout/stderr in a background task
                        tauri::async_runtime::spawn(async move {
                            use tauri_plugin_shell::process::CommandEvent;
                            while let Some(event) = rx.recv().await {
                                match event {
                                    CommandEvent::Stdout(line) => log::info!("[seth-server] {}", String::from_utf8_lossy(&line).trim()),
                                    CommandEvent::Stderr(line) => log::warn!("[seth-server] {}", String::from_utf8_lossy(&line).trim()),
                                    CommandEvent::Terminated(_) => break,
                                    _ => {}
                                }
                            }
                        });
                        log::info!("SETH sidecar spawned.");
                    }
                    Err(e) => {
                        log::error!("Failed to spawn sidecar: {}", e);
                        // Don't panic, maybe the user started it manually
                        app.manage(SidecarState(Mutex::new(None)));
                    }
                }
            } else {
                log::error!("Failed to resolve resource path for seth-server");
                app.manage(SidecarState(Mutex::new(None)));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building SETH");

    app.run(|app, event| {
        // Kill the sidecar when the app is about to exit
        if let tauri::RunEvent::ExitRequested { .. } = event {
            if let Some(state) = app.try_state::<SidecarState>() {
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(child) = guard.take() {
                        log::info!("Killing seth-server sidecar...");
                        let _ = child.kill();
                    }
                }
            }
        }
    });
}
