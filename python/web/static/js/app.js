"use strict";

let socket = null;
let currentSpeed = 25;
let streamActive = true;
let isRecording = false;
let toastTimer = null;
let wheelSpeeds = { rr: 0, fr: 0, fl: 0, rl: 0 };
let servoAngles = { 1: 110, 2: 50 };
let logAutoFollow = true;
let lastLogSignature = "";
let behaviorAlarmSignature = "";
let behaviorAlarmsById = new Map();
let alarmPage = 0;
const ALARM_PAGE_SIZE = 5;

const WHEEL_SIGN = { rr: -1, fr: -1, fl: 1, rl: 1 };
const wheelIds = {
    rr: { value: "valRR", display: "wRR", slider: "sliderRR", position: "wheel-rr" },
    fr: { value: "valFR", display: "wFR", slider: "sliderFR", position: "wheel-fr" },
    fl: { value: "valFL", display: "wFL", slider: "sliderFL", position: "wheel-fl" },
    rl: { value: "valRL", display: "wRL", slider: "sliderRL", position: "wheel-rl" },
};

function byId(id) {
    return document.getElementById(id);
}

function valueClass(value, baseClass) {
    return `${baseClass} ${value > 0 ? "positive" : value < 0 ? "negative" : "zero"}`;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function updateWheel(wheel, value) {
    const v = Number.parseInt(value, 10);
    const ids = wheelIds[wheel];
    if (!ids || Number.isNaN(v)) return;

    const valueEl = byId(ids.value);
    const displayEl = byId(ids.display);
    const positionEl = document.querySelector(`.${ids.position}`);

    valueEl.textContent = v;
    valueEl.className = valueClass(v, "wheel-val");
    displayEl.textContent = v;
    displayEl.className = valueClass(v, "wheel-speed");
    positionEl?.classList.toggle("active", v !== 0);

    wheelSpeeds[wheel] = Math.abs(v) < 6 ? 0 : v * WHEEL_SIGN[wheel];
}

function syncWheels() {
    socket?.emit("command", { action: "motor_raw", wheels: { ...wheelSpeeds } });
    showNotification("四轮速度已同步到电机");
}

function setAllWheels(value) {
    Object.keys(wheelIds).forEach((wheel) => {
        byId(wheelIds[wheel].slider).value = value;
        updateWheel(wheel, value);
    });
    syncWheels();
}

function resetWheels() {
    setAllWheels(0);
    showNotification("轮速已重置");
}

function initSocket() {
    if (typeof io !== "function") {
        setConnectionState(false, "通信组件未加载");
        return;
    }

    socket = io();
    socket.on("connect", () => setConnectionState(true, "系统已连接"));
    socket.on("disconnect", () => setConnectionState(false, "系统未连接"));
    socket.on("state_update", updateDataPanel);
    socket.on("pid_update", (data) => console.debug("PID updated", data));
    socket.on("threshold_update", handleThresholdUpdate);
    socket.on("capture_done", () => showNotification("截图已保存"));
    socket.on("record_done", (data) => {
        showNotification(`录像已保存，共 ${data.frames || 0} 帧`);
        setRecordState(false, 0);
    });
    socket.on("clean_done", (data) => {
        showNotification(`已清理 ${data.deleted || 0} 张调试图像`);
    });

    window.setInterval(fetchLogs, 2000);
    window.setInterval(fetchRecordingState, 1000);
}

function setConnectionState(connected, text) {
    byId("connDot").className = `dot ${connected ? "connected" : "disconnected"}`;
    byId("connText").textContent = text;
}

function isLogNearBottom(box) {
    return box.scrollHeight - box.scrollTop - box.clientHeight <= 28;
}

function updateLogFollowState(following) {
    logAutoFollow = following;
    const button = byId("logFollowBtn");
    const status = byId("logStatus");
    const statusText = byId("logStatusText");

    if (button) button.hidden = following;
    status?.classList.toggle("paused", !following);
    if (statusText) statusText.textContent = following ? "TAILING" : "PAUSED";
}

function resumeLogFollow() {
    const box = byId("logBox");
    if (!box) return;
    updateLogFollowState(true);
    box.scrollTop = box.scrollHeight;
}

function bindLogControls() {
    const box = byId("logBox");
    if (!box) return;

    box.addEventListener("scroll", () => {
        updateLogFollowState(isLogNearBottom(box));
    }, { passive: true });
    byId("logFollowBtn")?.addEventListener("click", resumeLogFollow);
}

async function fetchLogs() {
    if (!socket?.connected) return;
    try {
        const response = await fetch("/api/log");
        if (!response.ok) return;
        const logs = await response.json();
        if (!Array.isArray(logs) || logs.length === 0) return;
        const box = byId("logBox");
        const signature = logs.map((line) => `${line.time}\u0000${line.msg}`).join("\u0001");
        if (signature === lastLogSignature) return;

        const previousScrollTop = box.scrollTop;
        const shouldFollow = logAutoFollow || isLogNearBottom(box);
        box.innerHTML = logs.map((line) => (
            `<div class="log-line"><span class="log-time">${escapeHtml(line.time)}</span>` +
            `<span>${escapeHtml(line.msg)}</span></div>`
        )).join("");
        lastLogSignature = signature;

        if (shouldFollow) {
            box.scrollTop = box.scrollHeight;
            updateLogFollowState(true);
        } else {
            box.scrollTop = previousScrollTop;
        }
    } catch {
        // A temporary polling failure should not interrupt vehicle controls.
    }
}

async function fetchRecordingState() {
    if (!socket?.connected) return;
    try {
        const response = await fetch("/api/state");
        if (!response.ok) return;
        const data = await response.json();
        setRecordState(Boolean(data.recording), data.record_frames || 0);
    } catch {
        // State also arrives over WebSocket; polling is only a recording fallback.
    }
}

function updateDataPanel(data) {
    const state = data.state || "INIT";
    byId("stateValue").textContent = state;
    byId("fpsValue").textContent = Number(data.fps || 0).toFixed(1);
    byId("errorValue").textContent = Number(data.pid_error || 0).toFixed(1);
    byId("pidValue").textContent = Number(data.pid_output || 0).toFixed(3);
    byId("curveValue").textContent = data.is_curve ? "弯道" : "直道";
    byId("speedFactor").textContent = Number(data.speed_factor || 1).toFixed(2);
    byId("actionValue").textContent = data.action || "Stop";
    byId("inferTime").textContent = `${Number(data.infer_time_ms || 0).toFixed(0)} ms`;

    const threshold = Number(data.stop_threshold_cm ?? 20);
    const distance = data.distance_cm;
    const distanceEl = byId("distanceValue");
    if (distance !== undefined && distance !== null && distance >= 0) {
        distanceEl.textContent = `${Number(distance).toFixed(1)} cm`;
        distanceEl.className = `data-value${distance <= threshold ? " state-stop" : distance <= threshold * 2 ? " state-turn" : ""}`;
    } else {
        distanceEl.textContent = "--";
        distanceEl.className = "data-value";
    }
    byId("thresholdValue").textContent = `${threshold.toFixed(1)} cm`;
    syncThresholdUi(threshold);

    const servo = Array.isArray(data.servo_angle) ? data.servo_angle : [110, 50];
    byId("servoValue").textContent = `${servo[0]}° / ${servo[1]}°`;
    syncServoUi(1, servo[0]);
    syncServoUi(2, servo[1]);

    const stateEl = byId("stateValue");
    stateEl.className = "";
    if (state === "CRUISE" || state === "RUNNING") stateEl.classList.add("state-cruise");
    if (["TURN", "APPROACH", "PAUSE"].includes(state)) stateEl.classList.add("state-turn");
    if (state === "STOP") stateEl.classList.add("state-stop");

    byId("curveValue").className = `data-value${data.is_curve ? " state-turn" : ""}`;
    updateSigns(data.signs_detected || [], state);
    updateWheelTelemetry(data.wheel_speeds || {});
    updateModeButtons(data.mode || "idle");

    const preview = Boolean(data.preview_mode);
    byId("previewText").textContent = preview ? "PREVIEW" : "LIVE / ESP32";
    byId("previewBadge").title = preview ? "未连接 ESP32，当前为预览模式" : "ESP32 已连接";
    setRecordState(Boolean(data.recording), data.record_frames || 0);
    updateDebugUI(data);
}

function syncServoUi(channel, value) {
    const slider = byId(channel === 1 ? "servo1Slider" : "servo2Slider");
    const label = byId(channel === 1 ? "servo1Label" : "servo2Label");
    if (document.activeElement !== slider) slider.value = value;
    label.textContent = value;
    servoAngles[channel] = Number(value);
}

function updateSigns(signs, state) {
    const detected = new Set(signs);
    byId("signLeft")?.classList.toggle("detected", detected.has("left"));
    byId("signRight")?.classList.toggle("detected", detected.has("right"));
    byId("signTurnaround")?.classList.toggle("detected", detected.has("turnaround"));
    byId("signStop")?.classList.toggle("detected", state === "STOP" || detected.has("stop"));
}

function updateWheelTelemetry(speeds) {
    Object.keys(wheelIds).forEach((wheel) => {
        const value = Number(speeds[wheel] || 0);
        const el = byId(wheelIds[wheel].display);
        el.textContent = value;
        el.className = valueClass(value, "wheel-speed");
        document.querySelector(`.${wheelIds[wheel].position}`)?.classList.toggle("active", value !== 0);
    });
}

function updateModeButtons(mode) {
    document.querySelectorAll(".btn-mode").forEach((button) => button.classList.remove("active"));
    const buttonId = { idle: "btnIdle", manual: "btnManual", smart: "btnSmart" }[mode];
    if (buttonId) byId(buttonId).classList.add("active");
}

function setMode(mode) {
    socket?.emit("command", { action: "set_mode", mode });
    updateModeButtons(mode);
    const labels = { idle: "停止", manual: "手动驾驶", smart: "智能巡航" };
    showNotification(`已切换至${labels[mode]}模式`);
}

function motorCmd(direction) {
    socket?.emit("command", { action: "motor", direction, speed: currentSpeed });
}

function manualWheelsForDirection(direction) {
    const speed = Number.parseInt(currentSpeed, 10);
    const turnBoost = 5;
    if (direction === "forward") return { rr: -speed, fr: -speed, fl: speed, rl: speed };
    if (direction === "backward") return { rr: speed, fr: speed, fl: -speed, rl: -speed };
    if (direction === "left") {
        const rightSpeed = speed + turnBoost;
        return { rr: -rightSpeed, fr: -rightSpeed, fl: speed, rl: speed };
    }
    if (direction === "right") {
        const leftSpeed = speed + turnBoost;
        return { rr: -speed, fr: -speed, fl: leftSpeed, rl: leftSpeed };
    }
    return { rr: 0, fr: 0, fl: 0, rl: 0 };
}

function motorRawCmd(direction) {
    socket?.emit("command", {
        action: "motor_raw",
        wheels: manualWheelsForDirection(direction),
    });
}

function bindDriveControls() {
    document.querySelectorAll(".dpad-btn").forEach((button) => {
        const direction = button.dataset.direction;
        const press = (event) => {
            event.preventDefault();
            button.classList.add("pressed");
            motorCmd(direction);
        };
        const release = (event) => {
            event.preventDefault();
            button.classList.remove("pressed");
            if (direction !== "stop") motorCmd("stop");
        };
        button.addEventListener("pointerdown", press);
        button.addEventListener("pointerup", release);
        button.addEventListener("pointerleave", release);
        button.addEventListener("pointercancel", release);
    });
}

function updateSpeed(value) {
    currentSpeed = Number.parseInt(value, 10);
    byId("speedLabel").textContent = currentSpeed;
}

function setSpeed(value) {
    currentSpeed = value;
    byId("speedSlider").value = value;
    byId("speedLabel").textContent = value;
}

function previewStopThreshold(value) {
    byId("thresholdLabel").textContent = Number(value).toFixed(0);
}

function syncThresholdUi(value) {
    const slider = byId("thresholdSlider");
    if (document.activeElement !== slider) {
        slider.value = value;
        previewStopThreshold(value);
    }
}

function setStopThreshold(value) {
    byId("thresholdSlider").value = value;
    previewStopThreshold(value);
    applyStopThreshold();
}

function applyStopThreshold() {
    const threshold = Number(byId("thresholdSlider").value);
    socket?.emit("command", { action: "set_stop_threshold", threshold_cm: threshold });
    showNotification(`正在设置超声波停止距离：${threshold} cm`);
}

function handleThresholdUpdate(data) {
    if (data.ok) {
        syncThresholdUi(Number(data.threshold_cm));
        showNotification(`超声波停止距离已设为 ${Number(data.threshold_cm).toFixed(0)} cm`);
        return;
    }
    syncThresholdUi(Number(data.threshold_cm || 20));
    showNotification("设置失败：ESP32 固件需要升级");
}

// Debug模式控制
function toggleDebug(enabled) {
    socket?.emit("command", { action: "set_debug", enabled });
    const status = byId("debugStatus");
    if (status) {
        status.textContent = enabled ? "已开启" : "已关闭";
        status.className = `debug-status ${enabled ? "active" : ""}`;
    }
    showNotification(`调试模式已${enabled ? "开启" : "关闭"}`);
}

function cleanDebugImages() {
    if (!confirm("确定要删除所有调试图像吗？此操作不可恢复。")) {
        return;
    }
    socket?.emit("command", { action: "clean_debug" });
    showNotification("正在清理调试图像...");
}

function updateDebugUI(data) {
    const debugEnabled = Boolean(data.debug_mode);
    const toggle = byId("debugToggle");
    const status = byId("debugStatus");
    if (toggle) toggle.checked = debugEnabled;
    if (status) {
        status.textContent = debugEnabled ? "已开启" : "已关闭";
        status.className = `debug-status ${debugEnabled ? "active" : ""}`;
    }
}

function behaviorStateLabel(state) {
    return {
        idle: "等待任务",
        loading: "模型加载中",
        detecting: "实时检测中",
        completed: "检测完成",
        stopped: "检测已停止",
        error: "检测异常",
        offline: "推理服务离线",
    }[state] || "状态未知";
}

function setBehaviorBusy(busy) {
    byId("behaviorUploadBtn").disabled = busy;
    byId("behaviorLiveBtn").disabled = busy;
}

function startBehaviorStream() {
    const image = byId("behaviorStream");
    image.src = `/api/behavior/stream?t=${Date.now()}`;
    image.hidden = false;
    byId("behaviorEmpty").hidden = true;
}

async function uploadBehaviorVideo(file) {
    if (!file) return;
    const form = new FormData();
    form.append("video", file);
    setBehaviorBusy(true);
    showNotification(`正在上传并分析：${file.name}`);
    try {
        const response = await fetch("/api/behavior/upload", {
            method: "POST",
            body: form,
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "视频上传失败");
        }
        startBehaviorStream();
        showNotification("视频已导入，异常行为检测已启动");
    } catch (error) {
        showNotification(`检测启动失败：${error.message}`);
    } finally {
        setBehaviorBusy(false);
        byId("behaviorVideoInput").value = "";
    }
}

async function startBehaviorLive() {
    setBehaviorBusy(true);
    try {
        const response = await fetch("/api/behavior/live", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: 0 }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "实时检测启动失败");
        }
        startBehaviorStream();
        showNotification("电脑摄像头实时检测已启动");
    } catch (error) {
        showNotification(`实时检测启动失败：${error.message}`);
    } finally {
        setBehaviorBusy(false);
    }
}

async function stopBehaviorDetection() {
    try {
        await fetch("/api/behavior/stop", { method: "POST" });
        showNotification("异常行为检测停止指令已发送");
    } catch {
        showNotification("无法连接异常行为推理服务");
    }
}

function showBehaviorAlarm(eventId) {
    const event = behaviorAlarmsById.get(eventId);
    if (!event) return;
    const reasons = (event.reasons || []).map((reason) => reason.label).join(" / ");
    byId("latestAlertEmpty").hidden = true;
    byId("latestAlertContent").hidden = false;
    byId("latestAlertImage").src =
        `/api/behavior/alarms/${encodeURIComponent(event.id)}/image?t=${Date.now()}`;
    byId("latestAlertReason").textContent = reasons || "异常行为报警";
    byId("latestAlertTime").textContent =
        `${event.date} ${event.time} · FRAME ${event.frame_id}`;
}

function renderBehaviorAlarms(alarms) {
    const signature = alarms.map((event) => event.id).join("|");
    if (signature === behaviorAlarmSignature) return;
    behaviorAlarmSignature = signature;
    behaviorAlarmsById = new Map(alarms.map((event) => [event.id, event]));
    byId("behaviorAlarmCount").textContent = alarms.length;

    if (alarms.length === 0) {
        byId("behaviorAlarmList").innerHTML =
            '<div class="alert-timeline-empty">等待检测事件...</div>';
        byId("latestAlertEmpty").hidden = false;
        byId("latestAlertContent").hidden = true;
        byId("alarmPagination").hidden = true;
        alarmPage = 0;
        return;
    }

    const totalPages = Math.ceil(alarms.length / ALARM_PAGE_SIZE);
    if (alarmPage >= totalPages) alarmPage = totalPages - 1;
    if (alarmPage < 0) alarmPage = 0;

    renderAlarmPage(alarms);
    showBehaviorAlarm(alarms[0].id);
}

function renderAlarmPage(alarms) {
    if (!alarms) return;
    const totalPages = Math.ceil(alarms.length / ALARM_PAGE_SIZE);
    const start = alarmPage * ALARM_PAGE_SIZE;
    const page = alarms.slice(start, start + ALARM_PAGE_SIZE);

    byId("behaviorAlarmList").innerHTML = page.map((event) => {
        const reasons = (event.reasons || []).map((reason) => (
            `<span class="alarm-reason alarm-${escapeHtml(reason.code)}">` +
            `${escapeHtml(reason.label)} ` +
            `<b>${(Number(reason.confidence || 0) * 100).toFixed(0)}%</b></span>`
        )).join("");
        return (
            `<button type="button" class="alarm-event" data-alarm-id="${escapeHtml(event.id)}">` +
            `<span class="alarm-event-time">${escapeHtml(event.time)}</span>` +
            `<span class="alarm-event-main">${reasons}` +
            `<small>${escapeHtml(event.date)} · FRAME ${escapeHtml(event.frame_id)}</small></span>` +
            "</button>"
        );
    }).join("");
    byId("behaviorAlarmList").querySelectorAll(".alarm-event").forEach((button) => {
        button.addEventListener("click", () => showBehaviorAlarm(button.dataset.alarmId));
    });

    const pag = byId("alarmPagination");
    pag.hidden = totalPages <= 1;
    byId("alarmPrevBtn").disabled = alarmPage <= 0;
    byId("alarmNextBtn").disabled = alarmPage >= totalPages - 1;
    byId("alarmPageInfo").textContent = `${alarmPage + 1} / ${totalPages}`;
}

function alarmGoPage(delta) {
    const alarms = Array.from(behaviorAlarmsById.values());
    const totalPages = Math.ceil(alarms.length / ALARM_PAGE_SIZE);
    const next = alarmPage + delta;
    if (next < 0 || next >= totalPages) return;
    alarmPage = next;
    renderAlarmPage(alarms);
}

async function refreshBehaviorPanel() {
    try {
        const [statusResponse, alarmsResponse] = await Promise.all([
            fetch("/api/behavior/status"),
            fetch("/api/behavior/alarms"),
        ]);
        const status = await statusResponse.json();
        const alarmsData = await alarmsResponse.json();
        const alarms = Array.isArray(alarmsData)
            ? alarmsData
            : (alarmsData.alarms || []);

        const statusEl = byId("behaviorStatus");
        statusEl.classList.toggle(
            "alarm",
            Array.isArray(status.active_reasons) && status.active_reasons.length > 0,
        );
        statusEl.classList.toggle(
            "offline",
            status.state === "offline" || status.state === "error",
        );
        byId("behaviorStatusText").textContent = behaviorStateLabel(status.state);
        byId("behaviorSource").textContent =
            String(status.source_name || "NO SOURCE").toUpperCase();
        byId("behaviorFps").textContent = Number(status.fps || 0).toFixed(1);
        byId("behaviorFrame").textContent = Number(status.frame_id || 0);
        byId("behaviorProgressBar").style.width =
            `${Math.max(0, Math.min(100, Number(status.progress || 0)))}%`;

        if (status.source_name && byId("behaviorStream").hidden) {
            startBehaviorStream();
        }
        renderBehaviorAlarms(alarms);
    } catch {
        byId("behaviorStatus").classList.add("offline");
        byId("behaviorStatusText").textContent = "推理服务离线";
    }
}

function bindBehaviorControls() {
    byId("behaviorUploadBtn")?.addEventListener(
        "click",
        () => byId("behaviorVideoInput").click(),
    );
    byId("behaviorVideoInput")?.addEventListener(
        "change",
        (event) => uploadBehaviorVideo(event.target.files?.[0]),
    );
    byId("behaviorLiveBtn")?.addEventListener("click", startBehaviorLive);
    byId("behaviorStopBtn")?.addEventListener("click", stopBehaviorDetection);
    byId("alarmPrevBtn")?.addEventListener("click", () => alarmGoPage(-1));
    byId("alarmNextBtn")?.addEventListener("click", () => alarmGoPage(1));
    refreshBehaviorPanel();
    window.setInterval(refreshBehaviorPanel, 1000);
}

function updatePID() {
    const params = {
        kp: Number.parseInt(byId("kpSlider").value, 10) / 10000,
        ki: Number.parseInt(byId("kiSlider").value, 10) / 10000,
        kd: Number.parseInt(byId("kdSlider").value, 10) / 10000,
        smooth_alpha: Number.parseInt(byId("alphaSlider").value, 10) / 100,
        center_deadband_px: Number.parseInt(byId("deadbandSlider").value, 10),
    };

    byId("kpLabel").textContent = params.kp.toFixed(4);
    byId("kiLabel").textContent = params.ki.toFixed(4);
    byId("kdLabel").textContent = params.kd.toFixed(4);
    byId("alphaLabel").textContent = params.smooth_alpha.toFixed(2);
    byId("deadbandLabel").textContent = params.center_deadband_px;
    socket?.emit("command", { action: "set_pid", ...params });
}

function resetPID() {
    byId("kpSlider").value = 12;
    byId("kiSlider").value = 0;
    byId("kdSlider").value = 5;
    byId("alphaSlider").value = 30;
    byId("deadbandSlider").value = 20;
    updatePID();
    showNotification("PID 参数已恢复默认");
}

async function updateHSV() {
    const params = {
        h_min: Number.parseInt(byId("hminSlider").value, 10),
        h_max: Number.parseInt(byId("hmaxSlider").value, 10),
        s_min: Number.parseInt(byId("sminSlider").value, 10),
        v_min: Number.parseInt(byId("vminSlider").value, 10),
        adaptive: byId("adaptiveCheck").checked,
    };

    byId("hminLabel").textContent = params.h_min;
    byId("hmaxLabel").textContent = params.h_max;
    byId("sminLabel").textContent = params.s_min;
    byId("vminLabel").textContent = params.v_min;

    try {
        await fetch("/api/hsv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
    } catch {
        showNotification("HSV 参数同步失败");
    }
}

function captureFrame() {
    socket?.emit("command", { action: "capture" });
}

function toggleStream() {
    const image = byId("videoStream");
    const button = byId("toggleBtn");
    const videoState = byId("videoState");

    if (streamActive) {
        image.removeAttribute("src");
        button.textContent = "恢复视频";
        videoState.textContent = "FEED PAUSED";
        showNotification("视频流已暂停");
    } else {
        image.src = `/video_feed?t=${Date.now()}`;
        button.textContent = "暂停视频";
        videoState.textContent = "LIVE FEED";
        showNotification("视频流已恢复");
    }
    streamActive = !streamActive;
}

function showNotification(message) {
    const toast = byId("toast");
    const overlay = byId("videoOverlay");
    const overlayText = byId("overlayText");

    toast.textContent = message;
    toast.classList.add("visible");
    overlayText.textContent = message;
    overlay.style.opacity = "1";

    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
        toast.classList.remove("visible");
        overlay.style.opacity = "0";
    }, 2200);
}

function toggleRecord() {
    socket?.emit("command", { action: isRecording ? "record_stop" : "record_start" });
}

function setRecordState(recording, frames) {
    isRecording = recording;
    const button = byId("recordBtn");
    const info = byId("recordInfo");
    button.classList.toggle("recording", recording);
    button.innerHTML = `<span class="record-dot"></span>${recording ? "停止录制" : "开始录制"}`;
    info.textContent = recording ? `REC · ${frames} 帧` : "";
}

async function runDetection() {
    const button = byId("detectBtn");
    const resultEl = byId("detectResult");
    const imageWrap = byId("detectImage");
    const image = byId("detectImg");

    button.textContent = "检测中...";
    button.disabled = true;
    resultEl.innerHTML = '<span class="muted">正在运行单帧推理...</span>';

    try {
        const response = await fetch("/api/detect", { method: "POST" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "检测失败");

        const boxes = data.boxes || [];
        if (boxes.length === 0) {
            resultEl.innerHTML = '<span class="state-turn">当前画面未检测到目标</span>';
            imageWrap.hidden = true;
        } else {
            const colors = { stop: "#ff6b6b", left: "#51e6c2", right: "#68a7ff", turnaround: "#f2bd63" };
            resultEl.innerHTML = boxes.map((box) => {
                const color = colors[box.label] || "#f2bd63";
                return `<span class="result-tag" style="color:${color}">${escapeHtml(box.label)} ${(box.score * 100).toFixed(0)}%</span>`;
            }).join("");
            if (data.image_path) {
                image.src = `/api/detect/image?t=${Date.now()}`;
                imageWrap.hidden = false;
            }
        }
    } catch (error) {
        resultEl.innerHTML = `<span class="state-stop">检测失败：${escapeHtml(error.message)}</span>`;
        imageWrap.hidden = true;
    } finally {
        button.textContent = "运行检测";
        button.disabled = false;
    }
}

function updateServo(channel, value) {
    const angle = Number.parseInt(value, 10);
    servoAngles[channel] = angle;
    byId(channel === 1 ? "servo1Label" : "servo2Label").textContent = angle;
    socket?.emit("command", { action: "servo", channel, angle });
}

function setServo(channel, value) {
    byId(channel === 1 ? "servo1Slider" : "servo2Slider").value = value;
    updateServo(channel, value);
}

function bindKeyboardControls() {
    const keyMap = {
        w: "forward",
        ArrowUp: "forward",
        s: "backward",
        ArrowDown: "backward",
        a: "left",
        ArrowLeft: "left",
        d: "right",
        ArrowRight: "right",
        " ": "stop",
    };
    const activeKeys = new Set();

    const activeDirections = () => {
        return new Set([...activeKeys].map((key) => keyMap[key]).filter(Boolean));
    };

    const sendKeyboardCommand = () => {
        const directions = activeDirections();
        if (directions.has("stop")) {
            motorCmd("stop");
        } else if (directions.has("forward") && directions.has("right")) {
            motorRawCmd("right");
        } else if (directions.has("forward") && directions.has("left")) {
            motorRawCmd("left");
        } else if (directions.has("right")) {
            motorCmd("right");
        } else if (directions.has("left")) {
            motorCmd("left");
        } else if (directions.has("backward")) {
            motorCmd("backward");
        } else if (directions.has("forward")) {
            motorCmd("forward");
        } else {
            motorCmd("stop");
        }
    };

    document.addEventListener("keydown", (event) => {
        const direction = keyMap[event.key];
        if (!direction || event.target.matches("input, button")) return;
        event.preventDefault();
        if (!activeKeys.has(event.key)) {
            activeKeys.add(event.key);
            sendKeyboardCommand();
        }
    });

    document.addEventListener("keyup", (event) => {
        if (!keyMap[event.key]) return;
        activeKeys.delete(event.key);
        sendKeyboardCommand();
    });

    window.addEventListener("blur", () => {
        if (activeKeys.size > 0) motorCmd("stop");
        activeKeys.clear();
    });
}

document.addEventListener("DOMContentLoaded", () => {
    bindDriveControls();
    bindKeyboardControls();
    bindLogControls();
    bindBehaviorControls();
    initSocket();
});
