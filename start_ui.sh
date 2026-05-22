#!/bin/sh
# Start Streamlit on internal port 8501, auth proxy on public port 8502
streamlit run app.py \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &
STREAMLIT_PID=$!

python streamlit_auth_proxy.py &
PROXY_PID=$!

# If either process dies, kill both and exit
wait -n 2>/dev/null || true
kill $STREAMLIT_PID $PROXY_PID 2>/dev/null
wait
