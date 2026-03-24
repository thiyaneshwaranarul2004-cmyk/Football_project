import cv2
import numpy as np
from ultralytics import YOLO

cv2.setUseOptimized(True)

print("Loading model...")
yolo_model = YOLO("yolov8n.pt")
print("Model loaded.")

cap = cv2.VideoCapture("footballclip.mp4")

trackers = []
team_ids = []
player_ids = []

player_counter = {"Red": 0, "Blue": 0}

frame_count = 0
ball_position = None

team_possession = {"Red": 0, "Blue": 0}

player_paths = {}
player_distance = {}

# -----------------------------
def detect_team_color(player_crop):
    hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0,70,50), (10,255,255))
    blue_mask = cv2.inRange(hsv, (100,70,50), (140,255,255))

    return "Red" if cv2.countNonZero(red_mask) > cv2.countNonZero(blue_mask) else "Blue"

# -----------------------------
def compute_iou(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2

    xi1, yi1 = max(x1,x2), max(y1,y2)
    xi2, yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)

    inter = max(0,xi2-xi1)*max(0,yi2-yi1)
    union = w1*h1 + w2*h2 - inter

    return inter/union if union else 0

# -----------------------------
def draw_tv_ui(frame, red_pct, blue_pct, strategy_lines):
    h, w, _ = frame.shape

    # TOP BAR
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,60), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "RED", (50,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.putText(frame, "BLUE", (w-180,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    cv2.putText(frame, f"{red_pct}% | {blue_pct}%",
                (w//2 - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    # BOTTOM STRATEGY PANEL
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0,h-120), (w,h), (0,0,0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    y = h - 90
    for line in strategy_lines:
        cv2.putText(frame, line, (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)
        y += 25

# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,360))
    clean = frame.copy()

    frame_count += 1

    updated_boxes = []
    updated_teams = []
    updated_ids = []
    new_trackers = []

    # -------- TRACKERS --------
    for i, t in enumerate(trackers):
        ok, box = t.update(clean)

        if ok:
            x,y,w,h = map(int, box)

            updated_boxes.append((x,y,w,h))
            updated_teams.append(team_ids[i])
            updated_ids.append(player_ids[i])

            nt = cv2.legacy.TrackerCSRT_create()
            nt.init(clean,(x,y,w,h))
            new_trackers.append(nt)

    trackers = new_trackers
    team_ids = updated_teams
    player_ids = updated_ids

    # -------- YOLO --------
    if frame_count % 5 == 0:

        results = yolo_model(frame, imgsz=320, verbose=False)

        for r in results:
            for b in r.boxes:

                x1,y1,x2,y2 = map(int,b.xyxy[0])
                label = r.names[int(b.cls[0])]

                if label == "sports ball":
                    ball_position = (x1,y1,x2,y2)

                if label == "person":

                    w,h = x2-x1,y2-y1
                    new_box = (x1,y1,w,h)

                    if any(compute_iou(new_box, ob) > 0.5 for ob in updated_boxes):
                        continue

                    crop = clean[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    team = detect_team_color(crop)

                    player_counter[team] += 1
                    pid = f"{team}_P{player_counter[team]}"

                    t = cv2.legacy.TrackerCSRT_create()
                    t.init(clean,(x1,y1,w,h))

                    trackers.append(t)
                    team_ids.append(team)
                    player_ids.append(pid)

    # -------- POSSESSION --------
    possessor = None
    min_dist = 999999

    if ball_position:
        bx = (ball_position[0]+ball_position[2])//2
        by = (ball_position[1]+ball_position[3])//2

        for i,(x,y,w,h) in enumerate(updated_boxes):
            cx, cy = x+w//2, y+h//2
            d = (cx-bx)**2 + (cy-by)**2

            if d < min_dist:
                min_dist = d
                possessor = i

    if possessor is not None:
        team_possession[team_ids[possessor]] += 1

    total = sum(team_possession.values()) + 1
    red_pct = int(team_possession["Red"]/total*100)
    blue_pct = int(team_possession["Blue"]/total*100)

    # -------- PLAYER PERFORMANCE --------
    for i,(x,y,w,h) in enumerate(updated_boxes):

        cx,cy = x+w//2, y+h//2

        if i not in player_paths:
            player_paths[i] = []
            player_distance[i] = 0

        player_paths[i].append((cx,cy))

        if len(player_paths[i]) >= 2:
            x1,y1 = player_paths[i][-2]
            x2,y2 = player_paths[i][-1]

            dist = ((x2-x1)**2+(y2-y1)**2)**0.5
            player_distance[i] += dist

    # -------- AI STRATEGY --------
    best = max(player_distance, key=player_distance.get, default=None)
    worst = min(player_distance, key=player_distance.get, default=None)

    strategy_lines = []

    if red_pct > blue_pct + 10:
        strategy_lines.append("Blue: Defend & press more")
        strategy_lines.append("Red: Attack from wings")
    elif blue_pct > red_pct + 10:
        strategy_lines.append("Red: Fall back & defend")
        strategy_lines.append("Blue: Increase attacking")
    else:
        strategy_lines.append("Both: Control midfield & pass more")

    if best is not None and best < len(player_ids):
        strategy_lines.append(f"{player_ids[best]}: Strong - push forward")

    if worst is not None and worst < len(player_ids):
        strategy_lines.append(f"{player_ids[worst]}: Weak - reposition")

    # -------- DRAW PLAYERS --------
    for i,(x,y,w,h) in enumerate(updated_boxes):

        color = (0,0,255) if team_ids[i]=="Red" else (255,0,0)

        cv2.rectangle(clean,(x,y),(x+w,y+h),color,2)

        cv2.putText(clean,
                    player_ids[i],
                    (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(255,255,255),2)

    # -------- BALL --------
    if ball_position:
        x1,y1,x2,y2 = ball_position
        cv2.rectangle(clean,(x1,y1),(x2,y2),(0,255,255),3)
        cv2.putText(clean,"BALL",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    # -------- UI --------
    draw_tv_ui(clean, red_pct, blue_pct, strategy_lines)

    cv2.imshow("AI FOOTBALL ANALYSIS", clean)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()