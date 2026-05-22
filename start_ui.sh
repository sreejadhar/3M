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

# Wait for either process to exit, then kill both
# (wait -n is bash-only; this loop works in sh/dash)
while kill -0 $STREAMLIT_PID 2>/dev/null && kill -0 $PROXY_PID 2>/dev/null; do
  sleep 5
done
kill $STREAMLIT_PID $PROXY_PID 2>/dev/null
wait
