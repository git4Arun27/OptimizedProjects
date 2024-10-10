import torch
from flask import Flask, jsonify, request, render_template
from PIL import Image
import io
import pathlib
import mysql.connector
from flask_cors import CORS

# Fix pathlib for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the YOLO model
model = torch.hub.load(
    "C:\\Users\\50312\\.cache\\torch\\hub\\ultralytics_yolov5_master",
    "custom",
    path="models\\BestTest.pt",
    source="local",
    force_reload=True
)
print("Model loaded successfully")

# MySQL database connection
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Arun@2002",
    database="prosol",
    auth_plugin="mysql_native_password"
)
mycursor = mydb.cursor()


@app.route("/")
def homePage():
    return render_template("index2.html")


def getMaterialId(noun, modifier):
    if noun == "None" and modifier == "None":
        return -1

    query = "SELECT MATERIALID FROM MATERIALS WHERE NOUN=%s AND MODIFIER=%s"
    mycursor.execute(query, (noun, modifier))
    result = mycursor.fetchone()  # Fetch only one result

    if result:
        return result[0]
    else:
        return -1


def getAttributes(materialId):
    query = "SELECT ATTRIBUTES, VAL FROM ATTVALUES WHERE MODIFIERID=%s"
    mycursor.execute(query, (materialId,))
    result = mycursor.fetchall()
    return result


@app.route('/predict', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['file']

    # Convert the image file to a PIL Image object
    image = Image.open(io.BytesIO(image.read()))

    # Perform inference
    results = model(image)

    # Process the results
    nounlst=[]
    modifierlst=[]
    detections = []
    for result in results.xyxy[0]:
        print(model.names[int(result[5])])# Loop through the detections
        detected_class = model.names[int(result[5])].split("-")
        noun = detected_class[0]
        modifier = detected_class[1]
        if(noun in nounlst and modifier in modifierlst):
            print("values repeated")

        else:
            print(noun + "-" + modifier)
            detection_info = {
                "noun": noun,
                "modifier": modifier,
                "Matching percentage": float(result[4])
            }

            # Retrieve material ID from the database
            material_id = getMaterialId(noun, modifier)
            if material_id == -1:
                return jsonify({"Error": "Requested Material Not Found"}), 400
            nounlst.append(noun)
            modifierlst.append(modifier)
            # Retrieve attributes for the material
            attributes = getAttributes(material_id)
            attributes_dict = {i[0]: i[1] for i in attributes}

            detection_info["Attributes"] = attributes_dict
            detections.append(detection_info)

    return jsonify(detections) if detections else jsonify({"Log": "Please recapture or upload material"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
