const form = document.querySelector('#prediction-form');
const review = document.querySelector('#review');
const output = document.querySelector('#prediction');
const message = document.querySelector('#message');
const score = document.querySelector('#score');
const health = document.querySelector('#health');
const version = document.querySelector('#model-version');
let busy = false;
function clearResult() {
  output.textContent = '—'; score.textContent = 'Model score will appear here';
  message.classList.remove('error'); message.textContent = 'Review changed. Analyze again to update the result.';
  document.querySelector('#scenario').textContent = 'Enter a review, then analyze';
  document.querySelector('#prediction-details').textContent = 'Analyze a review to inspect this prediction.';
}
review.addEventListener('input', clearResult);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  review.value = button.dataset.example; clearResult(); review.focus();
}));
async function checkReady() {
  try {
    const response = await fetch('/ready');
    if (!response.ok) throw new Error('Model unavailable');
    const result = await response.json(); health.textContent = '● Model ready'; version.textContent = result.model_version;
  } catch (error) {
    health.textContent = 'Model unavailable'; health.classList.add('error'); version.textContent = 'Unavailable';
    message.textContent = 'Model service is unavailable. Try again after it is ready.'; message.classList.add('error');
  }
}
form.addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return;
  const text = review.value.trim();
  if (!text) { clearResult(); message.textContent = 'Please enter a review.'; message.classList.add('error'); return; }
  busy = true; const controls = [...form.elements]; controls.forEach(control => control.disabled = true);
  output.textContent = '…'; score.textContent = 'Analyzing…'; message.classList.remove('error');
  message.textContent = 'Running the pretrained model on CPU…';
  try {
    const response = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail) ? result.detail.map(item => item.msg).join('; ') : result.detail;
      throw new Error(detail || 'Prediction unavailable');
    }
    output.textContent = result.label === 'POSITIVE' ? 'Positive' : 'Negative';
    score.textContent = `Model score: ${(result.score * 100).toFixed(1)}%`;
    document.querySelector('#scenario').textContent = `${result.token_count} tokens · ${result.inference_ms} ms server inference`;
    message.textContent = 'Score for the selected label. It is not measured accuracy or a calibrated guarantee.';
    version.textContent = result.model_version; document.querySelector('#prediction-details').textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    output.textContent = '—'; score.textContent = 'No prediction'; message.textContent = error.message; message.classList.add('error');
  } finally { busy = false; controls.forEach(control => control.disabled = false); }
});
checkReady();
