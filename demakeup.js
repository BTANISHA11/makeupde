var SZ = 128; // Keep the size smaller for faster processing

function renderImage(file) {
    var reader = new FileReader();
    reader.onload = function (event) {
        var the_url = event.target.result;
        var img = new Image();
        img.onload = function () {
            var canvas = document.getElementById('input-canvas');
            var ctx = canvas.getContext('2d');
            canvas.width = SZ; // Resize image to a fixed size for consistency
            canvas.height = SZ;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            runModel(ctx);
        };
        img.src = the_url;
    };
    reader.readAsDataURL(file);
}

function renderOutput(outputData) {
    var data = outputData['convolution2d_11'];
    var ctx = document.getElementById('output-canvas').getContext('2d');
    var canvas = ctx.canvas;
    var imageData = ctx.createImageData(canvas.width, canvas.height);
    var j = 0;
    for (var i = 0; i < 3 * canvas.width * canvas.height; i += 3) {
        imageData.data[j++] = 255 * Math.min(1, Math.max(0, data[i + 0]));
        imageData.data[j++] = 255 * Math.min(1, Math.max(0, data[i + 1]));
        imageData.data[j++] = 255 * Math.min(1, Math.max(0, data[i + 2]));
        imageData.data[j++] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    // Apply a faster sharpening method or skip for speed
    applySimpleSharpening(ctx, imageData);
}

function flatten(imageData) {
    var data = imageData.data;
    var flat = new Float32Array(3 * imageData.width * imageData.height);
    var j = 0;
    for (var i = 0; i < data.length; i += 4) {
        flat[j++] = data[i + 0] / 255;
        flat[j++] = data[i + 1] / 255;
        flat[j++] = data[i + 2] / 255;
    }
    return flat;
}

function showSpin() {
    var opts = {
        lines: 10, // Reduced lines for less load
        length: 20,
        width: 10,
        radius: 30,
        scale: 0.4,
        corners: 1,
        color: '#000',
        opacity: 0.25,
        rotate: 0,
        direction: 1,
        speed: 1,
        trail: 50,
        fps: 20,
        zIndex: 2e9,
        className: 'spinner',
        top: '50%',
        left: '50%',
        shadow: false,
        hwaccel: false,
        position: 'absolute'
    };
    var target = document.getElementById('preview');
    var spinner = new Spinner(opts).spin(target);
    return spinner;
}

function runModel(ctx) {
    var imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    var spinner = showSpin();
    var flat = flatten(imageData);
    var gpu = document.getElementById('gpu');
    var model = new KerasJS.Model({
        filepaths: {
            model: 'model.json',
            weights: 'model_weights.buf',
            metadata: 'model_metadata.json'
        },
        gpu: gpu.checked
    });
    model.ready().then(() => {
        var inputData = {
            'input_1': flat
        };
        model.predict(inputData).then(outputData => {
            spinner.stop();
            renderOutput(outputData);
        });
    });
}

// Simpler sharpening method to improve speed
function applySimpleSharpening(ctx, imageData) {
    var width = imageData.width;
    var height = imageData.height;

    var tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    var tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);

    ctx.filter = 'contrast(1.2) brightness(1.1)'; // Simple CSS filter for speed
    ctx.drawImage(tempCanvas, 0, 0);
}
