(function () {
  "use strict";

  const API = window.MIRA_API_BASE || "";
  const messagesEl = document.getElementById("chatMessages");
  const chipsEl = document.getElementById("quickReplies");
  const inputEl = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const restartBtn = document.getElementById("restartBtn");
  const avatarWrap = document.getElementById("vaAvatar");
  const vaPanel = document.getElementById("vaPanel");
  const statusEl = document.getElementById("vaStatus");
  const emergencyBanner = document.getElementById("emergencyBanner");
  const stepEls = document.querySelectorAll(".va-step");

  const STEP_UI = {
    welcome: 0,
    symptom: 0,
    symptom_extra: 0,
    age_group: 0,
    duration: 1,
    fever_check: 1,
    trend: 1,
    severity: 2,
    red_flags: 2,
    impact: 2,
    medication: 2,
    risk: 2,
    aftercare: 3,
    done: 3,
  };

  const STATUS_LABELS = {
    idle: "Ready to help",
    thinking: "Thinking…",
    speaking: "Speaking",
    listening: "Listening…",
    success: "Consultation complete",
  };

  let busy = false;
  let scrollRaf = 0;
  let sessionPromise = null;
  let typingTimer = 0;

  function prefetchSession() {
    sessionPromise = api("/start", { method: "POST" }).catch(() => null);
  }

  function setVaState(state) {
    if (avatarWrap) avatarWrap.dataset.state = state;
    if (statusEl) statusEl.textContent = STATUS_LABELS[state] || "";
    if (vaPanel) {
      vaPanel.dataset.thinking = state === "thinking" ? "true" : "false";
      vaPanel.dataset.showWaves = state === "listening" ? "true" : "false";
    }
  }

  function updateSteps(step) {
    const idx = STEP_UI[step] ?? 0;
    stepEls.forEach((el, i) => {
      el.classList.remove("is-active", "is-done");
      if (i < idx) el.classList.add("is-done");
      else if (i === idx) el.classList.add("is-active");
    });
  }

  function scrollChat() {
    if (!messagesEl) return;
    if (scrollRaf) return;
    scrollRaf = requestAnimationFrame(() => {
      scrollRaf = 0;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }

  function formatBotText(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  function createMessage(text, role, extraClass, staggerIndex) {
    const div = document.createElement("div");
    div.className = `msg msg-${role}${extraClass ? " " + extraClass : ""}`;
    if (role === "bot") {
      div.innerHTML = formatBotText(text);
    } else {
      div.textContent = text;
    }
    if (staggerIndex > 0) {
      div.style.animationDelay = `${staggerIndex * 45}ms`;
    }
    return div;
  }

  function addMessage(text, role, extraClass, staggerIndex) {
    const div = createMessage(text, role, extraClass, staggerIndex ?? 0);
    messagesEl.appendChild(div);
    scrollChat();
    return div;
  }

  function clearChips() {
    if (chipsEl) chipsEl.innerHTML = "";
  }

  function renderChips(suggestions) {
    clearChips();
    if (!chipsEl || !suggestions || !suggestions.length) return;

    const frag = document.createDocumentFragment();
    suggestions.forEach((label, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "quick-chip";
      btn.textContent = label;
      btn.dataset.label = label;
      if (i > 0) btn.style.animationDelay = `${Math.min(i * 30, 150)}ms`;
      frag.appendChild(btn);
    });
    chipsEl.appendChild(frag);
  }

  if (chipsEl) {
    chipsEl.addEventListener("click", (e) => {
      const btn = e.target.closest(".quick-chip");
      if (!btn || busy) return;
      inputEl.value = btn.dataset.label || btn.textContent;
      onSend();
    });
  }

  function showEmergency(show) {
    if (emergencyBanner) emergencyBanner.hidden = !show;
  }

  function showBotDialog(data) {
    const msgs =
      data.messages && data.messages.length ? data.messages : [data.response];
    setVaState("speaking");

    const frag = document.createDocumentFragment();
    msgs.forEach((text, i) => {
      const isResult = data.condition && data.step === "aftercare" && i === 0;
      frag.appendChild(createMessage(text, "bot", isResult ? "msg-result" : "", i));
    });
    messagesEl.appendChild(frag);
    scrollChat();

    if (data.emergency) showEmergency(true);
    if (data.input_hint && inputEl) inputEl.placeholder = data.input_hint;

    renderChips(data.suggestions);

    if (data.done && data.condition) {
      setVaState("success");
      updateSteps("done");
      stepEls.forEach((el) => el.classList.add("is-done"));
    } else if (data.condition && data.step === "aftercare") {
      setVaState("success");
      updateSteps("aftercare");
    } else {
      setVaState("idle");
      if (data.step) updateSteps(data.step);
    }
  }

  function showTyping() {
    if (document.getElementById("typingIndicator")) return;
    const el = document.createElement("div");
    el.className = "msg msg-typing";
    el.id = "typingIndicator";
    el.innerHTML = "<span></span><span></span><span></span>";
    messagesEl.appendChild(el);
    scrollChat();
    setVaState("thinking");
  }

  function hideTyping() {
    clearTimeout(typingTimer);
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
  }

  async function api(path, options) {
    const res = await fetch(API + path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) throw new Error("Request failed");
    return res.json();
  }

  function startSession() {
    if (sessionPromise) {
      const p = sessionPromise;
      sessionPromise = null;
      return p;
    }
    return api("/start", { method: "POST" });
  }

  function sendMessage(text) {
    return api("/consult", {
      method: "POST",
      body: JSON.stringify({ message: text }),
    });
  }

  function handleBotReply(data) {
    hideTyping();
    showBotDialog(data);

    const locked = Boolean(data.done && data.condition);
    inputEl.disabled = locked;
    sendBtn.disabled = locked;
    busy = false;

    if (!locked) inputEl.focus();
    prefetchSession();
  }

  async function onSend() {
    const text = inputEl.value.trim();
    if (!text || busy) return;

    busy = true;
    sendBtn.disabled = true;
    clearChips();
    inputEl.value = "";

    addMessage(text, "user");
    setVaState("listening");

    typingTimer = window.setTimeout(showTyping, 120);

    try {
      const data = await sendMessage(text);
      handleBotReply(data);
    } catch {
      hideTyping();
      setVaState("idle");
      addMessage(
        "I couldn't reach the server. Make sure `python app.py` is running, then try again.",
        "bot"
      );
      busy = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  async function init() {
    messagesEl.innerHTML = "";
    clearChips();
    showEmergency(false);
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.placeholder = "Type your answer…";
    updateSteps("welcome");
    setVaState("thinking");

    try {
      const data = await startSession();
      hideTyping();
      showBotDialog(data);
    } catch {
      hideTyping();
      showBotDialog({
        messages: [
          "Hello! I'm Dr. MIRA.",
          "Tell me your main symptom when you're ready.",
        ],
        suggestions: ["Fever", "Cough", "Headache", "I'm ready"],
        step: "welcome",
      });
    }
    inputEl.focus();
    prefetchSession();
  }

  sendBtn.addEventListener("click", onSend);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  });
  restartBtn.addEventListener("click", () => {
    sessionPromise = null;
    init();
  });

  prefetchSession();
  init();
})();
