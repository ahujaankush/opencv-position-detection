from typing import Any, Dict
from flask import Flask, render_template, request, jsonify
import cv2
import color
import socket
from time import sleep, time

app = Flask(__name__)
HOST = "172.20.10.13"  # Server address
PORT = 8080           # Server port

@app.route('/')
def index():
    return "Nothing important"

@app.route('/colorblocks')
def colors():
    return render_template('index.html')

@app.route('/state', methods=['POST'])  # Make sure to specify POST method
def execute():
    # Handle the incoming JSON data
    client_order = request.json
    
    if client_order is None:
        return jsonify({"error": "No JSON data received"}), 400  # Return error if no JSON

    client_order = [item['color'] for item in client_order]
    
    # Open the camera
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        return jsonify({"error": "Error: Could not open camera"}), 500  # Return error if camera fails
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Error: Could not read frame from camera"}), 500  # Return error if no frame
    
    frame, objs = color.execute(frame)
    
    # Extract the coordinates based on sorted colors
    new_coords = []
    for element in client_order:
        for obj in objs:
            if obj[1] == element:
                new_coords.append(obj[0])
                break
    
    # Create a byte string with the coordinates
    result = [f"{int(pos.center[0])},{int(pos.center[1])},{idx}" for idx, pos in enumerate(new_coords)]

    # Create the socket connection
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        print("Connected to server")
        # Send the result to the server
        
        for res in result:
            client_socket.sendall(res.encode())
            print("Message sent")
            sleep(1)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to communicate with the server"}), 500  # Handle socket connection errors
    
    finally:
        client_socket.close()
        print("Connection closed")
    
    return jsonify({"status": "Data sent successfully"}), 200  # Send success response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
