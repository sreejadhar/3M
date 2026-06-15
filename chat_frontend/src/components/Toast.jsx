import { useApp } from '../state.jsx';

export default function ToastContainer() {
  const { toasts } = useApp();
  return (
    <div className="toast-container" id="toastContainer">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type}`}>{t.message}</div>
      ))}
    </div>
  );
}
