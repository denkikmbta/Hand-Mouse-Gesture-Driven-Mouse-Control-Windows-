import sys, time, math, ctypes, threading
from collections import deque
import numpy as np
import cv2
from mediapipe import solutions as mp_solutions

if sys.platform != "win32":
    raise RuntimeError("Windows-only build (SendInput).")

user32 = ctypes.windll.user32
SendInput = user32.SendInput

PUL = ctypes.POINTER(ctypes.c_ulong)
class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL))
class INPUT(ctypes.Structure):
    _fields_ = (("type", ctypes.c_ulong),
                ("mi", MOUSEINPUT))

MOUSEEVENTF_MOVE      = 0x0001
MOUSEEVENTF_ABSOLUTE  = 0x8000
MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP   = 0x0010
MOUSEEVENTF_WHEEL     = 0x0800

SCR_W = user32.GetSystemMetrics(0)
SCR_H = user32.GetSystemMetrics(1)
DC_TIME = user32.GetDoubleClickTime() / 1000.0
DC_DX = user32.GetSystemMetrics(36)
DC_DY = user32.GetSystemMetrics(37)
DC_DIST = max(12, int((DC_DX + DC_DY) // 2))

def _norm_abs(x, y):
    X = int(x * 65535 // max(SCR_W - 1, 1))
    Y = int(y * 65535 // max(SCR_H - 1, 1))
    return X, Y

def mouse_move_to(x, y):
    X, Y = _norm_abs(int(x), int(y))
    SendInput(1, ctypes.byref(INPUT(type=0, mi=MOUSEINPUT(X, Y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, None))), ctypes.sizeof(INPUT))

def mouse_left_down():
    SendInput(1, ctypes.byref(INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None))), ctypes.sizeof(INPUT))
def mouse_left_up():
    SendInput(1, ctypes.byref(INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None))), ctypes.sizeof(INPUT))
def mouse_left_click():
    mouse_left_down(); mouse_left_up()
def mouse_right_click():
    SendInput(1, ctypes.byref(INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, None))), ctypes.sizeof(INPUT))
    SendInput(1, ctypes.byref(INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, None))), ctypes.sizeof(INPUT))
def mouse_scroll(steps):
    SendInput(1, ctypes.byref(INPUT(type=0, mi=MOUSEINPUT(0, 0, int(120 * steps), MOUSEEVENTF_WHEEL, 0, None))), ctypes.sizeof(INPUT))

class LowPass:
    def __init__(self, x0=None, alpha=0.0): self.y = x0; self.a = alpha
    def set_alpha(self, a): self.a = float(a)
    def filter(self, x):
        if self.y is None:
            self.y = float(x); return self.y
        self.y = self.a * float(x) + (1.0 - self.a) * self.y
        return self.y

class OneEuro:
    def __init__(self, min_cutoff=1.7, beta=0.015, d_cutoff=1.8):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.xf = LowPass(); self.dxf = LowPass()
        self.prev = None
    @staticmethod
    def _alpha(cutoff, dt):
        if cutoff <= 0.0: return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    def filter(self, x, t):
        if self.prev is None:
            self.prev = t; self.xf.y = float(x); self.dxf.y = 0.0
            return float(x)
        dt = max(1e-4, t - self.prev); self.prev = t
        dx = (float(x) - self.xf.y) / dt
        ad = self._alpha(self.d_cutoff, dt); self.dxf.set_alpha(ad); dx_hat = self.dxf.filter(dx)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt); self.xf.set_alpha(a)
        return self.xf.filter(x)

class MedianFilter2D:
    def __init__(self, win=3):
        self.bx = deque(maxlen=win); self.by = deque(maxlen=win)
    def filter(self, x, y):
        self.bx.append(float(x)); self.by.append(float(y))
        return float(np.median(self.bx)), float(np.median(self.by))

class CameraWorker(threading.Thread):
    def __init__(self, index=0, backend="dshow", cap_w=1280, cap_h=720, fps=30):
        super().__init__(daemon=True)
        be = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
        self.cap = cv2.VideoCapture(index, be)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        self.lock = threading.Lock()
        self.latest = None
        self.stopped = False
    def run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005); continue
            with self.lock:
                self.latest = frame
    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()
    def release(self):
        self.stopped = True
        time.sleep(0.02)
        try: self.cap.release()
        except: pass

def v2(p):
    return np.array([float(p[0]), float(p[1])], dtype=float)

def angle_deg(a, b, c):
    ba = v2(a) - v2(b); bc = v2(c) - v2(b)
    nba = ba / (np.linalg.norm(ba) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    cosv = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def colinearity_score(a, b, c):
    ac = v2(c) - v2(a); bc = v2(c) - v2(b)
    nac = ac / (np.linalg.norm(ac) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    return float((np.dot(nac, nbc) + 1.0) * 0.5)

def is_finger_extended(mcp, pip, tip, thresh_deg=160.0):
    return angle_deg(mcp, pip, tip) >= thresh_deg

class PinchFSM:
    def __init__(self, on_thr, off_thr, stable_frames=3, tap_max=0.20, hold_min=0.55):
        self.on_thr = on_thr; self.off_thr = off_thr
        self.stable_frames = stable_frames
        self.tap_max = tap_max; self.hold_min = hold_min
        self.state = "IDLE"
        self.down_t = 0.0; self.down_pos = (0, 0)
        self.frames_below = 0; self.frames_above = 0
        self.cooldown_until = 0.0
    def update(self, dist, t, cur_pos, move_tol_px=22):
        ev = []
        if dist < self.on_thr:    self.frames_below += 1; self.frames_above = 0
        elif dist > self.off_thr: self.frames_above += 1; self.frames_below = 0
        if t < self.cooldown_until:
            return ev
        if self.state == "IDLE":
            if self.frames_below >= self.stable_frames:
                self.state = "DOWN"; self.down_t = t; self.down_pos = cur_pos
        elif self.state == "DOWN":
            dur = t - self.down_t
            moved = math.hypot(cur_pos[0] - self.down_pos[0], cur_pos[1] - self.down_pos[1]) > move_tol_px
            if self.frames_above >= self.stable_frames:
                if dur <= self.tap_max:
                    ev.append(("tap", self.down_pos))
                self.state = "IDLE"; self.frames_above = 0; self.frames_below = 0
                self.cooldown_until = t + 0.05
            else:
                if (dur >= self.hold_min) and (not moved):
                    ev.append(("hold_start", self.down_pos))
                    self.state = "HELD"
        elif self.state == "HELD":
            if self.frames_above >= self.stable_frames:
                ev.append(("release", self.down_pos))
                self.state = "IDLE"; self.frames_above = 0; self.frames_below = 0
                self.cooldown_until = t + 0.05
        return ev

class ClickScheduler:
    def __init__(self): self.queue = []
    @staticmethod
    def _click_fn(pos):
        def _fn():
            mouse_move_to(pos[0], pos[1])
            mouse_left_click()
        return _fn
    def schedule_double(self, pos, delay):
        now = time.time()
        self.queue.append((now, self._click_fn(pos)))
        self.queue.append((now + max(0.02, min(delay, 0.25)), self._click_fn(pos)))
    def tick(self):
        now = time.time()
        remain = []
        for t, fn in self.queue:
            if now >= t: fn()
            else: remain.append((t, fn))
        self.queue = remain

class HandMouseApp:
    def __init__(self):
        self.mirror = True
        self.invert_x = False
        self.show_preview = True

        self.proc_w, self.proc_h = 640, 360
        self.cam_w, self.cam_h, self.cam_fps = 1280, 720, 30
        self.cam_backend = "dshow"
        self.cam_index = 0

        self.max_hands = 1
        self.model_complexity = 0
        self.det_conf = 0.55
        self.trk_conf = 0.55

        self.median = MedianFilter2D(3)
        self.fx = OneEuro(1.7, 0.015, 1.8)
        self.fy = OneEuro(1.7, 0.015, 1.8)

        self.cursor_sens = 0.65
        self.cursor_gamma = 1.90
        self.precision = False
        self.precision_factor = 0.45

        self.K_PINCH_IDX = 0.20
        self.K_PINCH_MID = 0.23
        self.K_PINCH_RING= 0.24
        self.PINCH_OFF_FACTOR = 1.55

        self.TAP_MAX = 0.20
        self.HOLD_MIN = 0.55
        self.TAP_MOVE_TOL = 22

        self.doubletap_window = min(0.5, DC_TIME * 0.9)
        self.doubletap_dist   = DC_DIST
        self.last_tap_time = 0.0
        self.last_tap_pos = (0, 0)

        self.K_SCROLL_ON = 0.20
        self.SCROLL_GAIN = 0.08
        self.SCROLL_SMOOTH_TAU = 0.12
        self.SCROLL_RATE_MAX = 30.0
        self.SCROLL_RATE_DEAD = 0.6
        self.SCROLL_LOCK_HOLD = 3.0
        self.SCROLL_DEADZONE_PX = 25
        self.AUTO_SCROLL_BASE = 4.0
        self.AUTO_SCROLL_K = 0.14
        self.AUTO_SCROLL_STEP = 1

        self.use_calib = True
        self.calib_min = np.array([+1e9, +1e9], float)
        self.calib_max = np.array([-1e9, -1e9], float)
        self.autocal_on = False
        self.ac_samples = []
        self.AC_DUR = 3.0
        self.AC_MARGIN = 0.015
        self.ac_start = 0.0
        self.manual_on = False
        self.manual_step = 0
        self.manual_pts_raw = []

        self.left_drag = False
        self.scroll_mode = False
        self.prev_scroll_y = None
        self.scroll_dir = 0
        self.scroll_lock = False
        self.scroll_lock_start = 0.0
        self.last_autoscroll_t = 0.0
        self.autoscroll_accum = 0.0
        self.scroll_rate = 0.0
        self.scroll_accum = 0.0
        self.scroll_manual_last_t = None
        self.scroll_anchor_y = None

        self.dwell_on = True
        self.dwell_time = 0.65
        self.dwell_radius = 22
        self.dwell_cd = 0.35
        self.dwell_anchor = None
        self.dwell_started = None
        self.last_dwell_click = 0.0

        self.clicker = ClickScheduler()

        self.fsm_left = None
        self.fsm_right = None
        self.fsm_ring = None

    @staticmethod
    def draw_text(img, text, y, color=(255, 255, 255), scale=0.5, thick=1):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))
    @staticmethod
    def dist2(a, b): dx = a[0] - b[0]; dy = a[1] - b[1]; return dx*dx + dy*dy
    @staticmethod
    def hyst(active, d, on, off): return (d < on) if not active else not (d > off)

    def lmk(self, lm, i):
        x, y = lm[i].x, lm[i].y
        if self.mirror: x = 1.0 - x
        return np.array([x, y], float)

    @staticmethod
    def hand_scale(lm, w, h):
        p5 = np.array([lm[5].x * w, lm[5].y * h])
        p17 = np.array([lm[17].x * w, lm[17].y * h])
        return float(np.linalg.norm(p5 - p17)) + 1e-6

    def map_unit(self, p):
        if self.use_calib and (self.calib_max[0] > self.calib_min[0] + 1e-3) and (self.calib_max[1] > self.calib_min[1] + 1e-3):
            t = (p - self.calib_min) / (self.calib_max - self.calib_min); t = np.clip(t, 0, 1)
        else:
            t = np.clip(p, 0, 1)
        if self.invert_x: t[0] = 1.0 - t[0]
        return t

    def response_axis(self, u):
        sens = self.cursor_sens * (self.precision_factor if self.precision else 1.0)
        sens = float(np.clip(sens, 0.3, 1.2))
        gamma = float(np.clip(self.cursor_gamma, 1.0, 2.5))
        c = 0.5 + (u - 0.5) * sens
        x = (c - 0.5) * 2.0
        y = (x ** gamma) if x >= 0 else -((abs(x)) ** gamma)
        return float(np.clip(y * 0.5 + 0.5, 0.0, 1.0))

    def to_screen(self, u): return int(u[0] * SCR_W), int(u[1] * SCR_H)

    def run(self):
        cam = CameraWorker(self.cam_index, self.cam_backend, self.cam_w, self.cam_h, self.cam_fps); cam.start()
        hands = mp_solutions.hands.Hands(static_image_mode=False, max_num_hands=self.max_hands,
                                         model_complexity=self.model_complexity,
                                         min_detection_confidence=self.det_conf,
                                         min_tracking_confidence=self.trk_conf)
        print("Keys: q/ESC quit | h preview | m mirror | i invertX | p Precision | , . Sens | ; ' Gamma | a AutoCal | k ManualCal | Space confirm | r reset | [ ] scroll | 9/0 smooth | d Dwell")

        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.002)
                continue

            disp = cv2.flip(frame, 1) if self.mirror else frame
            proc = cv2.resize(disp, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            now = time.time()
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark

                tip_idx = self.lmk(lm, 8);  pip_idx = self.lmk(lm, 6)
                tip_th  = self.lmk(lm, 4)
                tip_mid = self.lmk(lm,12);  pip_mid = self.lmk(lm,10)
                tip_ring= self.lmk(lm,16);  pip_ring= self.lmk(lm,14)
                mcp_idx = self.lmk(lm, 5);  mcp_mid = self.lmk(lm, 9); mcp_ring = self.lmk(lm,13)

                idx_norm_raw = 0.8 * tip_idx + 0.2 * pip_idx

                if self.autocal_on:
                    self.ac_samples.append(idx_norm_raw.copy())
                    if now - self.ac_start > self.AC_DUR:
                        arr = np.array(self.ac_samples)
                        xs, ys = arr[:, 0], arr[:, 1]
                        lo = np.array([np.percentile(xs, 1),  np.percentile(ys, 1 )], float)
                        hi = np.array([np.percentile(xs, 99), np.percentile(ys, 99)], float)
                        rng = hi - lo
                        self.calib_min = np.clip(lo - self.AC_MARGIN * rng, 0, 1)
                        self.calib_max = np.clip(hi + self.AC_MARGIN * rng, 0, 1)
                        self.autocal_on = False; self.ac_samples.clear()
                    else:
                        pass

                u = self.map_unit(idx_norm_raw.copy())
                mxu, myu = self.median.filter(u[0], u[1])
                ax, ay = self.response_axis(mxu), self.response_axis(myu)
                x = self.fx.filter(ax, now); y = self.fy.filter(ay, now)
                mx, my = self.to_screen((x, y))
                mouse_move_to(mx, my)

                tip_idx_px = np.array([tip_idx[0] * self.proc_w, tip_idx[1] * self.proc_h])
                tip_mid_px = np.array([tip_mid[0] * self.proc_w, tip_mid[1] * self.proc_h])
                tip_ring_px= np.array([tip_ring[0]* self.proc_w, tip_ring[1]* self.proc_h])
                tip_th_px  = np.array([tip_th[0]  * self.proc_w, tip_th[1]  * self.proc_h])
                mcp_idx_px = np.array([mcp_idx[0]* self.proc_w, mcp_idx[1]* self.proc_h])
                mcp_mid_px = np.array([mcp_mid[0]* self.proc_w, mcp_mid[1]* self.proc_h])
                mcp_ring_px= np.array([mcp_ring[0]*self.proc_w, mcp_ring[1]*self.proc_h])
                pip_idx_px = np.array([pip_idx[0]* self.proc_w, pip_idx[1]* self.proc_h])
                pip_mid_px = np.array([pip_mid[0]* self.proc_w, pip_mid[1]* self.proc_h])
                pip_ring_px= np.array([pip_ring[0]*self.proc_w, pip_ring[1]*self.proc_h])

                d_tip_idx  = float(np.linalg.norm(tip_th_px - tip_idx_px))
                d_tip_mid  = float(np.linalg.norm(tip_th_px - tip_mid_px))
                d_tip_ring = float(np.linalg.norm(tip_th_px - tip_ring_px))

                align_idx  = colinearity_score(mcp_idx_px, tip_th_px, tip_idx_px)
                align_mid  = colinearity_score(mcp_mid_px, tip_th_px, tip_mid_px)
                align_ring = colinearity_score(mcp_ring_px, tip_th_px, tip_ring_px)

                ext_idx  = is_finger_extended(mcp_idx_px, pip_idx_px, tip_idx_px)
                ext_mid  = is_finger_extended(mcp_mid_px, pip_mid_px, tip_mid_px)
                ext_ring = is_finger_extended(mcp_ring_px, pip_ring_px, tip_ring_px)

                def effective_dist(d_tip, align, extended):
                    k_align = 1.0 - 0.35 * align
                    k_extend= 0.85 if extended else 1.00
                    return d_tip * k_align * k_extend
                d_eff_idx  = effective_dist(d_tip_idx,  align_idx,  ext_idx)
                d_eff_mid  = effective_dist(d_tip_mid,  align_mid,  ext_mid)
                d_eff_ring = effective_dist(d_tip_ring, align_ring, ext_ring)

                scale = self.hand_scale(lm, self.proc_w, self.proc_h)

                pin_idx_on  = self.K_PINCH_IDX  * scale; pin_idx_off  = self.K_PINCH_IDX  * self.PINCH_OFF_FACTOR * scale
                pin_mid_on  = self.K_PINCH_MID  * scale; pin_mid_off  = self.K_PINCH_MID  * self.PINCH_OFF_FACTOR * scale
                pin_ring_on = self.K_PINCH_RING * scale; pin_ring_off = self.K_PINCH_RING * self.PINCH_OFF_FACTOR * scale

                d_idx_mid = float(np.linalg.norm(tip_idx_px - tip_mid_px))
                scr_on = self.K_SCROLL_ON * scale
                scr_off= self.K_SCROLL_ON * self.PINCH_OFF_FACTOR * scale

                if (self.fsm_left is None) or (abs(self.fsm_left.on_thr - pin_idx_on) > 1e-6):
                    self.fsm_left  = PinchFSM(pin_idx_on,  pin_idx_off,  stable_frames=3, tap_max=self.TAP_MAX, hold_min=self.HOLD_MIN)
                if (self.fsm_right is None) or (abs(self.fsm_right.on_thr - pin_mid_on) > 1e-6):
                    self.fsm_right = PinchFSM(pin_mid_on,  pin_mid_off,  stable_frames=3, tap_max=0.18, hold_min=0.40)
                if (self.fsm_ring is None) or (abs(self.fsm_ring.on_thr - pin_ring_on) > 1e-6):
                    self.fsm_ring  = PinchFSM(pin_ring_on,  pin_ring_off, stable_frames=3, tap_max=0.18, hold_min=0.40)

                prev_scroll = self.scroll_mode
                self.scroll_mode = self.hyst(self.scroll_mode, d_idx_mid, scr_on, scr_off)
                if self.scroll_mode:
                    if not prev_scroll:
                        self.prev_scroll_y = None; self.scroll_dir = 0; self.scroll_lock = False
                        self.scroll_lock_start = now; self.last_autoscroll_t = now; self.autoscroll_accum = 0.0
                        self.scroll_rate = 0.0; self.scroll_accum = 0.0; self.scroll_manual_last_t = now
                        self.scroll_anchor_y = tip_idx_px[1]
                    if not self.scroll_lock:
                        if self.scroll_manual_last_t is None: self.scroll_manual_last_t = now
                        dtm = max(1e-3, now - self.scroll_manual_last_t); self.scroll_manual_last_t = now
                        dy = tip_idx_px[1] - (self.prev_scroll_y if self.prev_scroll_y is not None else tip_idx_px[1])
                        self.prev_scroll_y = tip_idx_px[1]
                        dy_per_s = dy / dtm
                        target_rate = -dy_per_s * self.SCROLL_GAIN
                        alpha = 1.0 - math.exp(-dtm / self.SCROLL_SMOOTH_TAU)
                        self.scroll_rate += alpha * (target_rate - self.scroll_rate)
                        if abs(self.scroll_rate) < self.SCROLL_RATE_DEAD: self.scroll_rate = 0.0
                        self.scroll_rate = float(np.clip(self.scroll_rate, -self.SCROLL_RATE_MAX, self.SCROLL_RATE_MAX))
                        self.scroll_accum += self.scroll_rate * dtm
                        steps = int(self.scroll_accum)
                        if steps != 0:
                            mouse_scroll(steps); self.scroll_accum -= steps
                    dy_from_anchor = tip_idx_px[1] - self.scroll_anchor_y
                    new_dir = 0
                    if abs(dy_from_anchor) > self.SCROLL_DEADZONE_PX:
                        new_dir = -1 if dy_from_anchor < 0 else +1
                    if new_dir != self.scroll_dir:
                        self.scroll_dir = new_dir; self.scroll_lock_start = now
                    if (self.scroll_dir != 0) and (not self.scroll_lock) and (now - self.scroll_lock_start >= self.SCROLL_LOCK_HOLD):
                        self.scroll_lock = True; self.last_autoscroll_t = now
                    if self.scroll_lock:
                        dt2 = now - self.last_autoscroll_t; self.last_autoscroll_t = now
                        overshoot = max(0.0, abs(dy_from_anchor) - self.SCROLL_DEADZONE_PX)
                        rate = self.AUTO_SCROLL_BASE + self.AUTO_SCROLL_K * overshoot
                        self.autoscroll_accum += rate * dt2
                        s = int(self.autoscroll_accum)
                        if s > 0:
                            mouse_scroll(self.scroll_dir * self.AUTO_SCROLL_STEP * s)
                            self.autoscroll_accum -= s
                else:
                    self.prev_scroll_y = None; self.scroll_lock = False; self.scroll_dir = 0

                ev_left = self.fsm_left.update(d_eff_idx, now, (mx, my), move_tol_px=self.TAP_MOVE_TOL)
                for kind, pos in ev_left:
                    if kind == "tap" and (not self.scroll_mode) and (not self.left_drag):
                        mouse_left_click()
                    elif kind == "hold_start" and (not self.scroll_mode) and (not self.left_drag):
                        self.clicker.schedule_double((mx, my), DC_TIME * 0.4)

                ev_right = self.fsm_right.update(d_eff_mid, now, (mx, my), move_tol_px=9999)
                for kind, _ in ev_right:
                    if kind == "tap" and (not self.left_drag) and (not self.scroll_mode):
                        mouse_right_click()

                ev_ring = self.fsm_ring.update(d_eff_ring, now, (mx, my), move_tol_px=9999)
                for kind, _ in ev_ring:
                    if kind == "hold_start" and (not self.scroll_mode):
                        if not self.left_drag:
                            mouse_left_down(); self.left_drag = True
                    elif kind == "release":
                        if self.left_drag:
                            mouse_left_up(); self.left_drag = False

                if self.dwell_on and (not self.left_drag) and (not self.scroll_mode):
                    cur = (mx, my)
                    if self.dwell_anchor is None:
                        self.dwell_anchor = cur; self.dwell_started = now
                    else:
                        if self.dist2(cur, self.dwell_anchor) <= (self.dwell_radius * self.dwell_radius):
                            if (now - (self.dwell_started or now) >= self.dwell_time) and (now - self.last_dwell_click >= self.dwell_cd):
                                mouse_left_click(); self.last_dwell_click = now
                                self.dwell_anchor = (cur[0] + self.dwell_radius * 2, cur[1] + self.dwell_radius * 2)
                                self.dwell_started = now
                        else:
                            self.dwell_anchor = cur; self.dwell_started = now

                if self.show_preview:
                    mp_solutions.drawing_utils.draw_landmarks(
                        disp, res.multi_hand_landmarks[0], mp_solutions.hands.HAND_CONNECTIONS
                    )
                    if (self.calib_max[0] < 1e8) and (self.calib_min[0] < 1e8):
                        H, W = disp.shape[:2]
                        x1f = float(self.calib_min[0]) * float(W); y1f = float(self.calib_min[1]) * float(H)
                        x2f = float(self.calib_max[0]) * float(W); y2f = float(self.calib_max[1]) * float(H)
                        x1 = int(max(0, min(W - 1, round(min(x1f, x2f)))))
                        y1 = int(max(0, min(H - 1, round(min(y1f, y2f)))))
                        x2 = int(max(0, min(W - 1, round(max(x1f, x2f)))))
                        y2 = int(max(0, min(H - 1, round(max(y1f, y2f)))))
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (60, 180, 60), 1, lineType=cv2.LINE_AA)

            self.clicker.tick()

            if self.show_preview:
                self.draw_text(disp, "Keys: q/ESC quit | h preview | m mirror | i invertX | p Precision | , . Sens | ; ' Gamma | a AutoCal | k ManualCal | Space confirm | r reset | [ ] scroll | 9/0 smooth | d Dwell", 26)
                info = f"Drag:{'ON' if self.left_drag else 'OFF'} | Sens:{self.cursor_sens:.2f}{' (P)' if self.precision else ''} | Gamma:{self.cursor_gamma:.2f} | Dwell:{'ON' if self.dwell_on else 'OFF'}"
                self.draw_text(disp, info, 50, (200,200,200), 0.55, 1)
                cv2.imshow("Hand Mouse - BEST v2 (EN)", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27): break
                elif key == ord('h'): self.show_preview = not self.show_preview
                elif key == ord('m'): self.mirror = not self.mirror
                elif key == ord('i'): self.invert_x = not self.invert_x
                elif key == ord('p'): self.precision = not self.precision
                elif key == ord(','): self.cursor_sens = max(0.3, self.cursor_sens - 0.05)
                elif key == ord('.'): self.cursor_sens = min(1.2, self.cursor_sens + 0.05)
                elif key == ord(';'): self.cursor_gamma = max(1.0, self.cursor_gamma - 0.05)
                elif key == ord('\''): self.cursor_gamma = min(2.5, self.cursor_gamma + 0.05)
                elif key == ord('['): self.SCROLL_GAIN = max(0.02, self.SCROLL_GAIN - 0.01)
                elif key == ord(']'): self.SCROLL_GAIN = min(0.30, self.SCROLL_GAIN + 0.01)
                elif key == ord('9'): self.SCROLL_SMOOTH_TAU = min(0.40, self.SCROLL_SMOOTH_TAU + 0.02)
                elif key == ord('0'): self.SCROLL_SMOOTH_TAU = max(0.05, self.SCROLL_SMOOTH_TAU - 0.02)
                elif key == ord('d'): self.dwell_on = not self.dwell_on
                elif key == ord('a'):
                    self.autocal_on = True; self.ac_samples.clear(); self.ac_start = time.time()
                elif key == ord('k'):
                    self.manual_on = True; self.manual_step = 1; self.manual_pts_raw.clear()
                elif key == ord(' '):
                    if self.manual_on:
                        self.manual_pts_raw.append(idx_norm_raw.copy())
                        if self.manual_step == 1:
                            self.manual_step = 2
                        elif self.manual_step == 2:
                            p1, p2 = self.manual_pts_raw[0], self.manual_pts_raw[1]
                            lo = np.minimum(p1, p2); hi = np.maximum(p1, p2); rng = hi - lo
                            lo = np.clip(lo - self.AC_MARGIN * rng, 0, 1)
                            hi = np.clip(hi + self.AC_MARGIN * rng, 0, 1)
                            self.calib_min, self.calib_max = lo, hi
                            self.manual_on = False; self.manual_step = 0
                elif key == ord('r'):
                    self.calib_min = np.array([+1e9, +1e9], float); self.calib_max = np.array([-1e9, -1e9], float)
            else:
                if cv2.waitKey(1) & 0xFF == ord('h'):
                    self.show_preview = True

        cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    HandMouseApp().run()
