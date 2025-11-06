#!/bin/bash
# Starts EmotionCV backend (FastAPI) and frontend (React or Streamlit)


set -e

BACKEND_DIR="/home/clyde/emotion-CV/backend"
FRONTEND_DIR="/home/clyde/emotion-CV/frontend"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT=8000
FRONTEND_PORT=8510
VENV_DIR="/mnt/wsl.localhost/Ubuntu-22.04/home/clyde/eNv"  

if [ -z "$OPENAI_API_KEY" ]; then
  echo "⚠️  OPENAI_API_KEY not set. Please export it or add it to your .env file."
fi

if [ -d "$VENV_DIR" ]; then
  echo "🔹 Activating virtual environment..."
  source "$VENV_DIR/bin/activate"
else
  echo "⚠️ Virtual environment not found at $VENV_DIR"
  echo "   Run 'python3 -m venv eNv && source eNv/bin/activate && pip install -r backend/requirements.txt'"
fi

echo "🔹 Cleaning up ports $BACKEND_PORT and $FRONTEND_PORT..."
lsof -i:$BACKEND_PORT -t | xargs -r kill -9
lsof -i:$FRONTEND_PORT -t | xargs -r kill -9

echo "🚀 Starting FastAPI backend on port $BACKEND_PORT..."
cd "$BACKEND_DIR"
uvicorn app:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!
cd ..

if [ -d "$FRONTEND_DIR" ]; then
  echo "🌐 Starting frontend on port $FRONTEND_PORT..."
  cd "$FRONTEND_DIR"

  if [ -f "package.json" ]; then
    npm run dev -- --port "$FRONTEND_PORT" &
  else
    streamlit run dashboard.py --server.port "$FRONTEND_PORT" &
  fi

  FRONTEND_PID=$!
  cd ..
else
  echo "⚠️ Frontend directory not found at $FRONTEND_DIR"
fi

echo "✅ EmotionCV is running!"
echo "   Backend: http://localhost:$BACKEND_PORT"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo
echo "Press [CTRL+C] to stop both servers."

trap "echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait

