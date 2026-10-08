const form = document.querySelector('#prediction-form');
const button = document.querySelector('#predict-button');
const health = document.querySelector('#health');
const output = document.querySelector('#prediction');
const message = document.querySelector('#message');
const version = document.querySelector('#model-version');
let busy = false;
function showVersion(value) { version.textContent = value; }
async function checkReady() {
  try {
    const response = await fetch('/ready');
    if (!response.ok) throw new Error('Model unavailable');
    const result = await response.json();
    health.textContent = '● Model ready';
    health.classList.remove('error');
    showVersion(result.model_version);
  } catch (error) {
    health.textContent = 'Model unavailable';
    health.classList.add('error');
    version.textContent = 'Unavailable';
    message.textContent = 'The model is unavailable. Check the service configuration and restart it.';
    message.classList.add('error');
  }
}
form.addEventListener('input', () => {
  if (!busy) {
    output.textContent = '—';
    message.classList.remove('error');
    message.textContent = 'Inputs changed. Predict again to update the estimate.';
    document.querySelector('#scenario').textContent = 'Choose your inputs, then predict';
    document.querySelector('#model-inputs').textContent = 'Make a prediction to inspect the inputs.';
  }
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  busy = true;
  // Keep the displayed inputs identical to the request while it is running.
  const values = new FormData(form);
  const controls = [...form.elements];
  controls.forEach(control => control.disabled = true);
  output.textContent = '…';
  message.classList.remove('error');
  message.textContent = 'Running the saved model…';
  const payload = Object.fromEntries(values.entries());
  ['hour', 'temperature_c', 'humidity_pct', 'windspeed_kmh'].forEach(key => payload[key] = Number(payload[key]));
  payload.holiday = values.has('holiday');
  try {
    const response = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const result = await response.json();
    if (!response.ok) {
      const text = Array.isArray(result.detail) ? result.detail.map(item => `${item.loc.slice(1).join('.')}: ${item.msg}`).join('; ') : result.detail;
      throw new Error(text || 'Prediction unavailable');
    }
    output.textContent = Math.round(result.predicted_rentals).toLocaleString();
    document.querySelector('#scenario').textContent = `${payload.date} · ${String(payload.hour).padStart(2, '0')}:00`;
    message.textContent = 'Estimated rentals during this hour. Rounded to the nearest whole rental.';
    showVersion(result.model_version);
    document.querySelector('#model-inputs').textContent = JSON.stringify(result.model_inputs, null, 2);
  } catch (error) {
    output.textContent = '—';
    message.textContent = error.message;
    message.classList.add('error');
  } finally {
    busy = false;
    controls.forEach(control => control.disabled = false);
  }
});
checkReady();
