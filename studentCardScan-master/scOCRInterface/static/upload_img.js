
/*  ==========================================
    SHOW UPLOADED IMAGE
* ========================================== */
var uploadBtn = document.getElementById('btn-upload');

function readURL(input) {
    $('#img').on('change', function () {
        readURL(input);
    });
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#imageResult')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
        uploadBtn.disabled = false;
    }else{
        uploadBtn.disabled = true;
    }
}


/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
var input = document.getElementById('img');
var infoArea = document.getElementById('img-label');

input.addEventListener('change', showFileName);

function showFileName(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea.textContent = 'File name: ' + fileName;
}

/*  ==========================================
    CAMERA
* ========================================== */

let camera_start = document.querySelector("#start-camera");
let camera_end = document.querySelector("#end-camera");
let video = document.querySelector("#video");
let click_button = document.querySelector("#click-photo");
let canvas = document.querySelector("#canvas");
var img_result = document.getElementById('imageResult');

camera_start.addEventListener('click', async function() {
   	let localstream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
	video.srcObject = localstream;
    document.getElementById("end-camera").style.display = "";
    document.getElementById("click-photo").style.display = "";
    document.getElementById("start-camera").style.display = "none";
    document.getElementById("video").style.display = "";
    document.getElementById("scan-photo").style.display = "none";
    uploadBtn.disabled = true;
    img_result.src = '#';
});

click_button.addEventListener('click', function() {
   	canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
   	let image_data_url = canvas.toDataURL('image/png');
	$('#imageResult').attr('src', image_data_url);

    $.ajax({
        url: "/studentCard/camera-image",
        type: "POST",
        contentType: "application/json",
        dataType: 'json',
        data: JSON.stringify(image_data_url),
        success: function(data) {
                successmessage = 'Data was succesfully captured';
                console.log(successmessage)
            },
            error: function(ts) {
                successmessage = 'Error';
                console.log(ts.responseText)
                window.location.reload();
            },

    });
    // setTimeout("location.reload(true);", 1000);

    document.getElementById("end-camera").style.display = "none";
    document.getElementById("click-photo").style.display = "none";
    document.getElementById("start-camera").style.display = "";
    document.getElementById("video").style.display = "none";
    document.getElementById("scan-photo").style.display = "";

    const stream = video.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => {
        track.stop();
    });
    video.srcObject = null;
});

camera_end.addEventListener('click', async function() {
    document.getElementById("end-camera").style.display = "none";
    document.getElementById("click-photo").style.display = "none";
    document.getElementById("start-camera").style.display = "";
    document.getElementById("video").style.display = "none";
    document.getElementById("scan-photo").style.display = "none";

   	const stream = video.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach((track) => {
        track.stop();
    });
    video.srcObject = null;
});

/*  ==========================================
    FUNCTIONS
* ========================================== */
