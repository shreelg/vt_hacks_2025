import os
import json
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import glob


app = FastAPI()
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/crops", StaticFiles(directory="crops"), name="crops")

# create folders
for folder in ["grids", "images", "results", "crops", "analytics", "monitoring"]:
    os.makedirs(folder, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join("parking_app", "templates"))

#CNN identifier
class SpotCNN(nn.Module):
    def __init__(self):
        super(SpotCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SpotCNN()
model_path = os.path.join(os.path.dirname(__file__), "spot_cnn.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

CLASS_NAMES = ["Free", "Occupied"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_spot(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]


# analytics func
def get_analytics_path(lot_name):
    return os.path.join("analytics", f"{lot_name}_history.json")

def load_analytics(lot_name):
    path = get_analytics_path(lot_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"checks": [], "spot_history": {}}

def save_analytics(lot_name, data):
    path = get_analytics_path(lot_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def calculate_spot_analytics(spot_history):
  
    if not spot_history:
        return {
            "total_checks": 0,
            "occupied_count": 0,
            "free_count": 0,
            "occupancy_rate": 0,
            "turnover_count": 0,
            "avg_occupied_duration": 0,
            "utilization_score": 0,
            "current_status": "Unknown"
        }
    
    total = len(spot_history)
    occupied = sum(1 for h in spot_history if h["status"] == "Occupied")
    free = total - occupied
    occupancy_rate = round((occupied / total) * 100, 1)
    current_status = spot_history[-1]["status"]
    
    # turn overs
    turnovers = 0
    for i in range(1, len(spot_history)):
        if spot_history[i]["status"] != spot_history[i-1]["status"]:
            turnovers += 1
    
    # avg duration in checks
    occupied_streaks = []
    current_streak = 0
    for h in spot_history:
        if h["status"] == "Occupied":
            current_streak += 1
        elif current_streak > 0:
            occupied_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        occupied_streaks.append(current_streak)
    
    avg_duration = round(sum(occupied_streaks) / len(occupied_streaks), 1) if occupied_streaks else 0
    
    # utilization score math
    turnover_rate = turnovers / total if total > 1 else 0
    utilization_score = round((occupancy_rate * 0.7) + (turnover_rate * 100 * 0.3), 1)
    
    return {
        "total_checks": total,
        "occupied_count": occupied,
        "free_count": free,
        "occupancy_rate": occupancy_rate,
        "turnover_count": turnovers,
        "avg_occupied_duration": avg_duration,
        "utilization_score": utilization_score,
        "current_status": current_status
    }

def find_trends(analytics_data):
    """Identify underutilized and overutilized spots"""
    spot_stats = {}
    
    for spot_id, history in analytics_data.get("spot_history", {}).items():
        stats = calculate_spot_analytics(history)
        spot_stats[spot_id] = stats
    
    if not spot_stats:
        return {
            "underutilized": [],
            "overutilized": [],
            "high_turnover": [],
            "low_turnover": [],
            "avg_occupancy": 0,
            "avg_turnover": 0
        }
    
    # calculate averages
    avg_occupancy = sum(s["occupancy_rate"] for s in spot_stats.values()) / len(spot_stats)
    avg_turnover = sum(s["turnover_count"] for s in spot_stats.values()) / len(spot_stats)
    
    underutilized = [
        {"spot": int(sid), "occupancy_rate": stats["occupancy_rate"], "utilization_score": stats["utilization_score"]}
        for sid, stats in spot_stats.items()
        if stats["occupancy_rate"] < avg_occupancy * 0.7 and stats["total_checks"] >= 5
    ]
    underutilized.sort(key=lambda x: x["occupancy_rate"])
    
    overutilized = [
        {"spot": int(sid), "occupancy_rate": stats["occupancy_rate"], "utilization_score": stats["utilization_score"]}
        for sid, stats in spot_stats.items()
        if stats["occupancy_rate"] > avg_occupancy * 1.3 and stats["total_checks"] >= 5
    ]
    overutilized.sort(key=lambda x: x["occupancy_rate"], reverse=True)
    
    high_turnover = [
        {"spot": int(sid), "turnover_count": stats["turnover_count"]}
        for sid, stats in spot_stats.items()
        if stats["turnover_count"] > avg_turnover * 1.5 and stats["total_checks"] >= 5
    ]
    high_turnover.sort(key=lambda x: x["turnover_count"], reverse=True)
    
    low_turnover = [
        {"spot": int(sid), "turnover_count": stats["turnover_count"]}
        for sid, stats in spot_stats.items()
        if stats["turnover_count"] < avg_turnover * 0.5 and stats["total_checks"] >= 5
    ]
    low_turnover.sort(key=lambda x: x["turnover_count"])
    
    return {
        "underutilized": underutilized[:10],
        "overutilized": overutilized[:10],
        "high_turnover": high_turnover[:10],
        "low_turnover": low_turnover[:10],
        "avg_occupancy": round(avg_occupancy, 1),
        "avg_turnover": round(avg_turnover, 1)
    }


def process_image(image_path, lot_name):
    """Process a single image and return results"""
    grid_path = os.path.join("grids", f"{lot_name}.json")
    
    if not os.path.exists(grid_path):
        return None
    
    with open(grid_path) as f:
        spots = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        return None
        
    crop_dir = os.path.join("crops", lot_name)
    os.makedirs(crop_dir, exist_ok=True)

    free_count = 0
    total = len(spots)
    spot_status = []
    spot_images = []
    timestamp = datetime.now().isoformat()

    analytics_data = load_analytics(lot_name)
    
    for i, (x1, y1, x2, y2) in enumerate(spots):
        crop = img[y1:y2, x1:x2]
        crop_filename = f"spot_{i}.jpg"
        crop_path = os.path.join(crop_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        label = predict_spot(crop)
        spot_status.append(label)
        spot_images.append(f"/crops/{lot_name}/{crop_filename}")

        # updating spot history
        spot_id = str(i)
        if spot_id not in analytics_data["spot_history"]:
            analytics_data["spot_history"][spot_id] = []
        analytics_data["spot_history"][spot_id].append({
            "timestamp": timestamp,
            "status": label
        })
        
        color = (0, 255, 0) if label == "Free" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        spot_number = str(i + 1)

        text_size = cv2.getTextSize(spot_number, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(img, spot_number, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if label == "Free":
            free_count += 1

    empty_spots = round(100 * (free_count / total), 2)
    taken_spots = round(100 * ((total - free_count) / total))

    analytics_data["checks"].append({
        "timestamp": timestamp,
        "free_count": free_count,
        "occupied_count": total - free_count,
        "total": total
    })

    # Limit history
    for spot_id in analytics_data["spot_history"]:
        if len(analytics_data["spot_history"][spot_id]) > 1000:
            analytics_data["spot_history"][spot_id] = analytics_data["spot_history"][spot_id][-1000:]
    
    save_analytics(lot_name, analytics_data)

    # Save annotated image
    filename = os.path.basename(image_path)
    annotated_path = os.path.join("results", filename)
    cv2.imwrite(annotated_path, img)

    # Create grid visualization
    grid_size = int(total ** 0.5) + 1
    cell_size = 60
    padding = 5
    grid_img = 255 * np.ones((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)

    for idx, status in enumerate(spot_status):
        row, col = divmod(idx, grid_size)
        x1, y1 = col * cell_size + padding, row * cell_size + padding
        x2, y2 = (col + 1) * cell_size - padding, (row + 1) * cell_size - padding

        color = (0, 255, 0) if status == "Free" else (0, 0, 255)
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), (50, 50, 50), 1)

        text = str(idx + 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(grid_img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    grid_path = os.path.join("results", f"{lot_name}_grid.jpg")
    cv2.imwrite(grid_path, grid_img)

    return {
        "free_count": free_count,
        "occupied_count": total - free_count,
        "total": total,
        "empty_spots": empty_spots,
        "taken_spots": taken_spots,
        "annotated_image": f"/results/{filename}",
        "grid_map": f"/results/{lot_name}_grid.jpg",
        "spot_images": spot_images,
        "spot_status": spot_status,
        "timestamp": timestamp
    }


# routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/upload")
async def admin_upload(file: UploadFile = File(...), lot_name: str = Form(...)):
    image_path = os.path.join("images", file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(image_path)
    spots = []

    display_w = 1200
    scale = display_w / img.shape[1]
    display_img = cv2.resize(img, (display_w, int(img.shape[0] * scale)))

    while True:
        rect = cv2.selectROI("Select Spot", display_img, showCrosshair=True, fromCenter=False)
        if rect == (0, 0, 0, 0):
            break
        x, y, w, h = rect
        spots.append((int(x/scale), int(y/scale), int((x+w)/scale), int((y+h)/scale)))

    cv2.destroyAllWindows()

    grid_path = os.path.join("grids", f"{lot_name}.json")
    with open(grid_path, "w") as f:
        json.dump(spots, f, indent=2)

    return {"message": f"Saved {len(spots)} spots for lot {lot_name}"}

@app.get("/student", response_class=HTMLResponse)
async def student_page(request: Request):
    return templates.TemplateResponse("student.html", {"request": request})

@app.get("/lots")
async def list_lots():
    lots = [f.replace(".json", "") for f in os.listdir("grids") if f.endswith(".json")]
    return {"lots": lots}


@app.get("/monitoring/images/{lot_name}")
async def get_monitoring_images(lot_name: str):
    """Get list of images available for monitoring"""
    monitoring_path = os.path.join("monitoring", lot_name)
    if not os.path.exists(monitoring_path):
        return JSONResponse({"images": [], "message": f"No images found for {lot_name}"})
    
    image_files = glob.glob(os.path.join(monitoring_path, "*.jpg")) + \
                  glob.glob(os.path.join(monitoring_path, "*.png"))
    
    images = [os.path.basename(f) for f in sorted(image_files)]
    return {"images": images, "total": len(images)}


@app.post("/monitoring/analyze")
async def analyze_monitoring_images(lot_name: str = Form(...)):
    """Analyze all images in the monitoring folder for a lot"""
    monitoring_path = os.path.join("monitoring", lot_name)
    
    if not os.path.exists(monitoring_path):
        return JSONResponse({
            "error": f"Monitoring folder not found for {lot_name}",
            "message": f"Please create folder: {monitoring_path}"
        }, status_code=404)
    
    grid_path = os.path.join("grids", f"{lot_name}.json")
    if not os.path.exists(grid_path):
        return JSONResponse({
            "error": f"Grid not configured for {lot_name}",
            "message": "Please configure the parking grid first in admin panel"
        }, status_code=404)
    
    image_files = glob.glob(os.path.join(monitoring_path, "*.jpg")) + \
                  glob.glob(os.path.join(monitoring_path, "*.png"))
    
    if not image_files:
        return JSONResponse({
            "error": "No images found",
            "message": f"Please add images to: {monitoring_path}"
        }, status_code=404)
    
    results = []
    for image_path in sorted(image_files):
        result = process_image(image_path, lot_name)
        if result:
            result["filename"] = os.path.basename(image_path)
            results.append(result)
    
    return JSONResponse({
        "lot_name": lot_name,
        "total_images": len(image_files),
        "processed": len(results),
        "results": results
    })


@app.get("/monitoring/latest/{lot_name}")
async def get_latest_analysis(lot_name: str):
    """Get the most recent analysis for a lot"""
    monitoring_path = os.path.join("monitoring", lot_name)
    
    if not os.path.exists(monitoring_path):
        return JSONResponse({"error": "Monitoring folder not found"}, status_code=404)
    
    image_files = glob.glob(os.path.join(monitoring_path, "*.jpg")) + \
                  glob.glob(os.path.join(monitoring_path, "*.png"))
    
    if not image_files:
        return JSONResponse({"error": "No images found"}, status_code=404)
    
    # Get most recent image
    latest_image = max(image_files, key=os.path.getmtime)
    result = process_image(latest_image, lot_name)
    
    if result:
        result["filename"] = os.path.basename(latest_image)
        return JSONResponse(result)
    else:
        return JSONResponse({"error": "Failed to process image"}, status_code=500)


@app.get("/monitoring/cycle/{lot_name}/{index}")
async def cycle_image_analysis(lot_name: str, index: int):
    """Get analysis for a specific image by index - cycles through images"""
    monitoring_path = os.path.join("monitoring", lot_name)
    
    if not os.path.exists(monitoring_path):
        return JSONResponse({
            "error": "Monitoring folder not found",
            "message": f"Please create folder: monitoring/{lot_name}/"
        }, status_code=404)
    
    image_files = sorted(glob.glob(os.path.join(monitoring_path, "*.jpg")) + \
                        glob.glob(os.path.join(monitoring_path, "*.png")))
    
    if not image_files:
        return JSONResponse({
            "error": "No images found",
            "message": f"Please add images to: monitoring/{lot_name}/"
        }, status_code=404)
    
    total_images = len(image_files)
    
    
    current_index = index % total_images
    image_path = image_files[current_index]
    
    result = process_image(image_path, lot_name)
    
    if result:
        result["filename"] = os.path.basename(image_path)
        result["current_index"] = current_index
        result["total_images"] = total_images
        result["next_index"] = (current_index + 1) % total_images
        return JSONResponse(result)
    else:
        return JSONResponse({"error": "Failed to process image"}, status_code=500)


@app.post("/student/check")
async def student_check(lot_name: str = Form(...)):
 
    
    monitoring_path = os.path.join("monitoring", lot_name)
    
    if not os.path.exists(monitoring_path):
        return JSONResponse({
            "error": f"Monitoring folder not found for {lot_name}",
            "message": f"Please create folder: monitoring/{lot_name}/ and add images"
        }, status_code=404)
    
    image_files = glob.glob(os.path.join(monitoring_path, "*.jpg")) + \
                  glob.glob(os.path.join(monitoring_path, "*.png"))
    
    if not image_files:
        return JSONResponse({
            "error": "No images found",
            "message": f"Please add images to: monitoring/{lot_name}/"
        }, status_code=404)
    
    # get most recent image
    image_path = max(image_files, key=os.path.getmtime)

    result = process_image(image_path, lot_name)
    
    if result:
        return JSONResponse(result)
    else:
        return JSONResponse({"error": "Failed to process image"}, status_code=500)


@app.get("/analytics/{lot_name}")
async def get_analytics(lot_name: str):
    analytics_data = load_analytics(lot_name)
    
    spot_analytics = {}
    for spot_id, history in analytics_data.get("spot_history", {}).items():
        spot_analytics[spot_id] = calculate_spot_analytics(history)
    
    if spot_analytics:
        avg_occupancy = sum(s["occupancy_rate"] for s in spot_analytics.values()) / len(spot_analytics)
        avg_turnover = sum(s["turnover_count"] for s in spot_analytics.values()) / len(spot_analytics)
        avg_duration = sum(s["avg_occupied_duration"] for s in spot_analytics.values()) / len(spot_analytics)
        avg_utilization = sum(s["utilization_score"] for s in spot_analytics.values()) / len(spot_analytics)
    else:
        avg_occupancy = avg_turnover = avg_duration = avg_utilization = 0
    
    trends = find_trends(analytics_data)
    
    return JSONResponse({
        "spot_analytics": spot_analytics,
        "averages": {
            "avg_occupancy_rate": round(avg_occupancy, 1),
            "avg_turnover": round(avg_turnover, 1),
            "avg_occupied_duration": round(avg_duration, 1),
            "avg_utilization_score": round(avg_utilization, 1)
        },
        "trends": trends,
        "total_checks": len(analytics_data.get("checks", []))
    })