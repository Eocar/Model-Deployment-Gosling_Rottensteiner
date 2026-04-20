async function loadModel() {
  const response = await fetch('./model.json');
  if (!response.ok) {
    throw new Error('Unable to load model.json. Run phase1_train.py first.');
  }
  return response.json();
}

function predict(model, values) {
  let score = model.intercept;
  model.coefficients.forEach((coef, idx) => {
    score += coef * values[idx];
  });
  return score;
}

function buildForm(featureNames) {
  const form = document.getElementById('predict-form');
  form.innerHTML = '';
  featureNames.forEach((name) => {
    const label = document.createElement('label');
    label.textContent = name;

    const input = document.createElement('input');
    input.type = 'number';
    input.step = '0.01';
    input.name = name;
    input.required = true;
    input.value = '0';

    label.appendChild(input);
    form.appendChild(label);
  });
}

(async () => {
  const model = await loadModel();
  buildForm(model.feature_names);

  document.getElementById('predict').addEventListener('click', () => {
    const values = model.feature_names.map((featureName) => {
      const input = document.querySelector(`input[name="${featureName}"]`);
      return Number(input.value || 0);
    });

    const prediction = predict(model, values);
    document.getElementById('result').textContent = `Predicted quality: ${prediction.toFixed(2)}`;
  });
})();
