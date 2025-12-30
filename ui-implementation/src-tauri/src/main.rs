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

// Read API port from backend/.env file
fn read_api_port_from_env(backend_dir: &std::path::Path) -> u16 {
    let env_file = backend_dir.join(".env");
    if let Ok(content) = std::fs::read_to_string(&env_file) {
        for line in content.lines() {
            if line.starts_with("ROAMPAL_API_PORT=") {
                if let Some(port_str) = line.strip_prefix("ROAMPAL_API_PORT=") {
                    if let Ok(port) = port_str.trim().parse::<u16>() {
                        println!("[read_api_port] Found port {} in .env file", port);
                        return port;
                    }
                }
            }
        }
    }
    println!("[read_api_port] Using default port 8765");
    8765 // Default to PROD port
}

// Read data directory name from backend/.env file
fn read_data_dir_from_env(backend_dir: &std::path::Path) -> String {
    let env_file = backend_dir.join(".env");
    if let Ok(content) = std::fs::read_to_string(&env_file) {
        for line in content.lines() {
            if line.starts_with("ROAMPAL_DATA_DIR=") {
                if let Some(dir_name) = line.strip_prefix("ROAMPAL_DATA_DIR=") {
                    let dir_name = dir_name.trim();
                    if !dir_name.is_empty() {
                        println!("[read_data_dir] Found data dir '{}' in .env file", dir_name);
                        return dir_name.to_string();
                    }
                }
            }
        }
    }
    println!("[read_data_dir] Using default data dir 'Roampal'");
    "Roampal".to_string() // Default to PROD data dir
}

// v0.2.9: Check if port is already in use (backend survived refresh)
fn is_port_in_use(port: u16) -> bool {
    use std::net::TcpListener;
    TcpListener::bind(("127.0.0.1", port)).is_err()
}

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

    // Read API port and data dir from .env file
    let api_port = read_api_port_from_env(&backend_dir);
    let data_dir = read_data_dir_from_env(&backend_dir);

    // v0.2.9: Check if port is already in use (backend survived window refresh)
    // This fixes the 120-second timeout when user presses Ctrl+R
    if is_port_in_use(api_port) {
        println!("[start_backend] Port {} already in use - backend survived refresh, reconnecting", api_port);
        return Ok(format!("Backend already running on port {}", api_port));
    }

    println!("[start_backend] Python exe: {:?}", python_exe);
    println!("[start_backend] Main.py: {:?}", main_py);
    println!("[start_backend] API port: {}", api_port);
    println!("[start_backend] Data dir: {}", data_dir);

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

        // Backend logging handled by Python's RotatingFileHandler in main.py
        // No need for stdout/stderr file logging (prevents unbounded log growth)
        let mut child_cmd = Command::new(&python_exe);
        child_cmd
            .arg(&main_py)
            .current_dir(backend_root)
            .env("PYTHONPATH", pythonpath)
            .env("ROAMPAL_API_PORT", api_port.to_string())
            .env("ROAMPAL_DATA_DIR", &data_dir)
            .creation_flags(CREATE_NO_WINDOW);

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

        println!("[start_backend] Backend process spawned successfully on port {}", api_port);
        *backend = Some(child);
        Ok(format!("Backend started from {} on port {}", main_py.display(), api_port))
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err("Only Windows is supported".to_string())
    }
}

// v0.2.8: Exit app command - kills backend and exits cleanly
#[tauri::command]
fn exit_app(backend_state: State<BackendProcess>, app_handle: tauri::AppHandle) -> Result<String, String> {
    println!("[exit_app] Exit requested, killing backend process...");

    // Kill backend process
    if let Ok(mut backend) = backend_state.0.lock() {
        if let Some(mut child) = backend.take() {
            let _ = child.kill();
            let _ = child.wait(); // Wait for process to fully terminate
            println!("[exit_app] Backend process killed");
        }
    }

    // Exit the app
    app_handle.exit(0);
    Ok("Exiting...".to_string())
}

// Check if backend is responding
#[tauri::command]
async fn check_backend() -> Result<bool, String> {
    // Get the current executable directory to read .env file for port
    let exe_dir = std::env::current_exe()
        .map_err(|e| format!("Failed to get exe path: {}", e))?
        .parent()
        .ok_or("Failed to get exe directory")?
        .to_path_buf();
    let backend_dir = exe_dir.join("backend");
    let api_port = read_api_port_from_env(&backend_dir);

    let url = format!("http://localhost:{}/health", api_port);
    println!("[check_backend] Checking {}", url);

    match reqwest::get(&url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false)
    }
}

fn run_mcp_backend() -> ! {
    // NOTE: MCP uses stdio for JSON-RPC protocol
    // All debug output MUST go to stderr to avoid corrupting the protocol
    eprintln!("[MCP] Starting Roampal MCP server (headless mode)...");

    let exe_dir = std::env::current_exe()
        .expect("Failed to get executable path")
        .parent()
        .expect("Failed to get executable directory")
        .to_path_buf();

    let python_exe = exe_dir.join("binaries").join("python").join("python.exe");
    let main_py = exe_dir.join("backend").join("main.py");
    let backend_dir = exe_dir.join("backend");

    eprintln!("[MCP] Python: {:?}", python_exe);
    eprintln!("[MCP] Main.py: {:?}", main_py);
    eprintln!("[MCP] Working dir: {:?}", backend_dir);

    if !python_exe.exists() {
        eprintln!("[MCP] ERROR: Python executable not found at {:?}", python_exe);
        std::process::exit(1);
    }

    if !main_py.exists() {
        eprintln!("[MCP] ERROR: main.py not found at {:?}", main_py);
        std::process::exit(1);
    }

    #[cfg(target_os = "windows")]
    {
        // NOTE: Don't use CREATE_NO_WINDOW for MCP mode - it breaks stdio pipes
        // MCP requires proper stdin/stdout for JSON-RPC communication
        let mut child = Command::new(&python_exe)
            .arg(&main_py)
            .arg("--mcp")
            .current_dir(&backend_dir)
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("Failed to start MCP server");

        let status = child.wait().expect("MCP server wait failed");
        std::process::exit(status.code().unwrap_or(1));
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut child = Command::new(&python_exe)
            .arg(&main_py)
            .arg("--mcp")
            .current_dir(&backend_dir)
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .expect("Failed to start MCP server");

        let status = child.wait().expect("MCP server wait failed");
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn main() {
    // Check for MCP mode before launching GUI
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--mcp") {
        run_mcp_backend();
    }

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

            // v0.2.8: X button no longer kills backend - use Settings > Exit Roampal for clean shutdown
            // This allows the app to close quickly while backend stays ready for quick reopen
            main_window.on_window_event(move |event| {
                match event {
                    tauri::WindowEvent::CloseRequested { .. } => {
                        println!("[Cleanup] Window close requested - backend will keep running");
                        println!("[Cleanup] Use Settings > Exit Roampal for full shutdown");
                        // Don't kill backend here - let exit_app command handle it
                    }
                    tauri::WindowEvent::Destroyed => {
                        // Only kill backend on actual window destruction (app force quit)
                        println!("[Cleanup] Window destroyed, killing backend process...");
                        if let Ok(mut backend) = backend_clone.lock() {
                            if let Some(mut child) = backend.take() {
                                let _ = child.kill();
                                let _ = child.wait();
                                println!("[Cleanup] Backend process killed");
                            }
                        }
                    }
                    _ => {}
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
            check_backend,
            exit_app
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}