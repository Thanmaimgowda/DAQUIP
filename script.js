document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const spinner = document.getElementById('loading-spinner');

    form.addEventListener('submit', function(event) {
        spinner.style.display = 'block';
    });
});
