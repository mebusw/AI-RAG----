// ecosystem.config.js
module.exports = {
  apps : [{
    name   : "streamlit-app", // A unique name for your application
    script : "streamlit",      // The command to execute (streamlit)
    args   : "run your_app.py --server.port 8501 --server.enableCORS false", // Arguments for streamlit
    interpreter: "python3",  // Specify the Python interpreter
    exec_mode: "fork",       // Run in fork mode (important for Streamlit)
    instances: 1,            // Number of instances (usually 1 for Streamlit unless behind a load balancer)
    cwd    : "/path/to/your/streamlit/app", // The working directory of your app
    env: {
      NODE_ENV: "production",
      // Add any environment variables your app needs (e.g., API keys)
      // MY_API_KEY: "your_secret_key"
    },
    error_file: "/var/log/streamlit-app-error.log", // Log file for errors
    out_file: "/var/log/streamlit-app-out.log",     // Log file for standard output
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    // Optional: Restart options
    autorestart: true,
    watch: false, // Set to true for development, false for production to avoid restarts on file changes
    max_memory_restart: "2G" // Restart if memory usage exceeds this (e.g., "2G", "500M")
  }]
};

//pm2 start ecosystem.config.js