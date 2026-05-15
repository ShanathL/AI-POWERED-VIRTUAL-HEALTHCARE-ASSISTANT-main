(function () {
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
    duration: 1,
    trend: 1,
    severity: 2,
    impact: 2,
    risk: 2,
    aftercare: 3,
    done: 3,
  };

  let busy = false;

  function setVaState(state) {
    if (avatarWrap) avatarWrap.dataset.state = state;
    const labels = {
      idle: "Ready to help",
      thinking: "Thinking…",
      speaking: "Speaking",
      listening: "Listening…",
      success: "Consultation complete",
    };
    if (statusEl) statusEl.textContent = labels[state] || "";
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
    if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function formatBotText(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  function addMessage(text, role, extraClass) {
    const div = document.createElement("div");
    div.className = `msg msg-${role}${extraClass ? " " + extraClass : ""}`;
    if (role === "bot") {
      div.innerHTML = formatBotText(text);
    } else {
      div.textContent = text;
    }
    messagesEl.appendChild(div);
    scrollChat();
    return div;
  }

  function clearChips() {
    if (chipsEl) chipsEl.innerHTML = "";
  }

  function renderChips(suggestions, onPick) {
    clearChips();
    if (!chipsEl || !suggestions || !suggestions.length) return;

    suggestions.forEach((label) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "quick-chip";
      btn.textContent = label;
      btn.addEventListener("click", () => {
        if (busy) return;
        onPick(label);
      });
      chipsEl.appendChild(btn);
    });
  }

  function showEmergency(show) {
    if (emergencyBanner) {
      emergencyBanner.hidden = !show;
    }
  }

  async function delay(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  async function showBotDialog(data) {
    const msgs = data.messages && data.messages.length ? data.messages : [data.response];
    setVaState("speaking");

    for (let i = 0; i < msgs.length; i++) {
      if (i > 0) await delay(420);
      const isResult = data.done && data.condition && i === 0;
      addMessage(msgs[i], "bot", isResult ? "msg-result" : "");
    }

    if (data.emergency) showEmergency(true);
    if (data.input_hint && inputEl) inputEl.placeholder = data.input_hint;

    renderChips(data.suggestions, (label) => {
      inputEl.value = label;
      onSend();
    });

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
    const el = document.createElement("div");
    el.className = "msg msg-typing";
    el.id = "typingIndicator";
    el.innerHTML = "<span></span><span></span><span></span>";
    messagesEl.appendChild(el);
    scrollChat();
    setVaState("thinking");
  }

  function hideTyping() {
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

  async function startSession() {
    return api("/start", { method: "POST" });
  }

  async function sendMessage(text) {
    return api("/consult", {
      method: "POST",
      body: JSON.stringify({ message: text }),
    });
  }

  async function handleBotReply(data) {
    hideTyping();
    await showBotDialog(data);

    if (data.done && data.condition) {
      inputEl.disabled = true;
      sendBtn.disabled = true;
    } else {
      inputEl.disabled = false;
      sendBtn.disabled = false;
    }

    busy = false;
    if (!inputEl.disabled) inputEl.focus();
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
    await delay(180);
    showTyping();

    try {
      const data = await sendMessage(text);
      await handleBotReply(data);
    } catch {
      hideTyping();
      setVaState("idle");
      addMessage(
        "I couldn't reach the server. Make sure `python app.py` is running, then try again.",
        "bot"
      );
      busy = false;
      sendBtn.disabled = false;
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
      await showBotDialog(data);
    } catch {
      hideTyping();
      await showBotDialog({
        messages: [
          "Hello! I'm Dr. MIRA.",
          "Tell me your main symptom when you're ready.",
        ],
        suggestions: ["Fever", "Cough", "Headache", "I'm ready"],
        step: "welcome",
      });
    }
    inputEl.focus();
  }

  sendBtn.addEventListener("click", onSend);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  });
  restartBtn.addEventListener("click", () => init());

  init();
})();
