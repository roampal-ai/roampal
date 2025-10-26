#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::process::{Command, Child};
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use futures_util::stream::TryStreamExt;

#[derive(Debug, Serialize, Deserialize)]
struct FileContent {
    path: String,
    content: String,
}

// Open file in VS Code
#[tauri::command]
fn open_in_vscode(path: String) -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "code", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        Command::new("code")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(format!("Opened {} in VS Code", path))
}

// Open folder in VS Code
#[tauri::command]
fn open_folder_in_vscode(path: String) -> Result<String, String> {
    Command::new("code")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(format!("Opened folder {} in VS Code", path))
}

// Read file from filesystem
#[tauri::command]
fn read_file(path: String) -> Result<FileContent, String> {
    let content = std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())?;
    Ok(FileContent { path, content })
}

// Write file to filesystem
#[tauri::command]
fn write_file(path: String, content: String) -> Result<String, String> {
    std::fs::write(&path, content)
        .map_err(|e| e.to_string())?;
    Ok(format!("File saved: {}", path))
}

// List files in directory
#[tauri::command]
fn list_files(path: String) -> Result<Vec<String>, String> {
    let entries = std::fs::read_dir(&path)
        .map_err(|e| e.to_string())?;

    let mut files = Vec::new();
    for entry in entries {
        if let Ok(entry) = entry {
            if let Ok(name) = entry.file_name().into_string() {
                files.push(name);
            }
        }
    }
    Ok(files)
}

// Run git command
#[tauri::command]
fn run_git_command(args: Vec<String>) -> Result<String, String> {
    let output = Command::new("git")
        .args(&args)
        .output()
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// Stream LLM responses for instant feedback
#[tauri::command]
async fn stream_llm_response(
    window: tauri::Window,
    prompt: String,
    model: Option<String>
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let model_name = model.unwrap_or_else(|| "mistral-nemo:12b-instruct-2407-q4_0".to_string());

    let response = client
        .post("http://localhost:11434/api/generate")
        .json(&serde_json::json!({
            "model": model_name,
            "prompt": prompt,
            "stream": true
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let stream = response.bytes_stream();
    let reader = tokio_util::io::StreamReader::new(
        stream.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    );
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        match buf_reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(response_text) = json.get("response").and_then(|r| r.as_str()) {
                        window.emit("llm-chunk", response_text).ok();
                    }
                    // Check if done
                    if json.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                        break;
                    }
                }
            }
            Err(_) => break,
        }
    }

    Ok(())
}

// Backend process state
struct BackendProcess(Arc<Mutex<Option<Child>>>);

// Start the Python backend
#[tauri::command]
fn start_backend(app_handle: tauri::AppHandle, backend_state: State<BackendProcess>) -> Result<String, String> {
    println!("[start_backend] Command invoked");

    let mut backend = backend_state.0.lock().unwrap();

    // Check if already running
    if let Some(ref mut child) = *backend {
        if let Ok(None) = child.try_wait() {
            println!("[start_backend] Backend already running");
            return Ok("Backend already running".to_string());
        }
    }

    // Get the current executable directory
    let exe_dir = std::env::current_exe()
        .map_err(|e| format!("Failed to get exe path: {}", e))?
        .parent()
        .ok_or("Failed to get exe directory")?
        .to_path_buf();

    println!("[start_backend] Exe dir: {:?}", exe_dir);

    // List directory contents for debugging
    println!("[start_backend] Listing contents of exe_dir:");
    if let Ok(entries) = std::fs::read_dir(&exe_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  - {:?}", entry.path());
            }
        }
    } else {
        println!("  [ERROR] Failed to read directory");
    }

    // Resources should be in the same directory as the executable
    let python_exe = exe_dir.join("binaries").join("python").join("python.exe");
    let backend_dir = exe_dir.join("backend");
    let main_py = backend_dir.join("main.py");
    let backend_root = &backend_dir;

    println!("[start_backend] Python exe: {:?}", python_exe);
    println!("[start_backend] Main.py: {:?}", main_py);

    // Check if main.py exists with more details
    println!("[start_backend] Checking main.py existence...");
    match std::fs::metadata(&main_py) {
        Ok(metadata) => {
            println!("[start_backend] main.py metadata: is_file={}, len={}", metadata.is_file(), metadata.len());
        }
        Err(e) => {
            println!("[start_backend] main.py metadata error: {}", e);
        }
    }

    // Check if files exist
    if !python_exe.exists() {
        return Err(format!("Python executable not found at: {:?}", python_exe));
    }
    if !main_py.exists() {
        return Err(format!("main.py not found at: {:?}", main_py));
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        println!("[start_backend] Spawning Python process...");
        // Set PYTHONPATH to include current directory (.) and backend_root
        let pythonpath = format!(".;{}", backend_root.display());

        // Try to create log files, but don't fail if we can't
        let log_dir = exe_dir.join("logs");
        let mut child_cmd = Command::new(&python_exe);
        child_cmd
            .arg(&main_py)
            .current_dir(backend_root)
            .env("PYTHONPATH", pythonpath)
            .creation_flags(CREATE_NO_WINDOW);

        // Attempt to set up logging, but continue without it if it fails
        match std::fs::create_dir_all(&log_dir) {
            Ok(_) => {
                let stdout_log = log_dir.join("backend_stdout.log");
                let stderr_log = log_dir.join("backend_stderr.log");

                match (std::fs::File::create(&stdout_log), std::fs::File::create(&stderr_log)) {
                    (Ok(stdout_file), Ok(stderr_file)) => {
                        println!("[start_backend] Logs will be written to:");
                        println!("  stdout: {:?}", stdout_log);
                        println!("  stderr: {:?}", stderr_log);
                        child_cmd.stdout(stdout_file).stderr(stderr_file);
                    }
                    _ => {
                        println!("[start_backend] Warning: Could not create log files, running without file logging");
                    }
                }
            }
            Err(e) => {
                println!("[start_backend] Warning: Could not create logs directory ({}), running without file logging", e);
            }
        }

        let child = child_cmd
            .spawn()
            .map_err(|e| {
                let error_msg = format!("Failed to start backend: {}", e);
                // Check for permission denied (Error 13)
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("{}\n\nThis is likely caused by antivirus blocking Python.\nTry:\n1. Right-click Roampal.exe → Properties → Unblock\n2. Add folder to Windows Defender exclusions\n3. Run as Administrator", error_msg)
                } else {
                    error_msg
                }
            })?;

        println!("[start_backend] Backend process spawned successfully");
        *backend = Some(child);
        Ok(format!("Backend started from {} (logs in {:?})", main_py.display(), log_dir))
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err("Only Windows is supported".to_string())
    }
}

// Check if backend is responding
#[tauri::command]
async fn check_backend() -> Result<bool, String> {
    match reqwest::get("http://localhost:8000/health").await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false)
    }
}

fn main() {
    tauri::Builder::default()
        .manage(BackendProcess(Arc::new(Mutex::new(None))))
        .setup(|app| {
            let main_window = app.get_window("main").unwrap();
            main_window.set_title("Roampal - Your Private Intelligence")?;
            main_window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                width: 1400,
                height: 900,
            }))?;

            // Set up cleanup handler for app exit
            let backend_state = app.state::<BackendProcess>();
            let backend_clone = Arc::clone(&backend_state.0);

            main_window.on_window_event(move |event| {
                if let tauri::WindowEvent::Destroyed = event {
                    println!("[Cleanup] Window destroyed, killing backend process...");
                    if let Ok(mut backend) = backend_clone.lock() {
                        if let Some(mut child) = backend.take() {
                            let _ = child.kill();
                            println!("[Cleanup] Backend process killed");
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_in_vscode,
            open_folder_in_vscode,
            read_file,
            write_file,
            list_files,
            run_git_command,
            stream_llm_response,
            start_backend,
            check_backend
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}