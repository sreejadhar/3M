import { useAppState } from '../state.jsx';

export default function ToastContainer() {
  const { toasts } = useAppState();
  return (
    <div id="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type}`}>
          {t.message}
        </div>
      ))}
    </div>
  );
}
