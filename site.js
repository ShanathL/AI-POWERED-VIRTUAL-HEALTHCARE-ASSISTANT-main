(function () {
  const siteHeader = document.getElementById('siteHeader');
  const headerActions = document.getElementById('headerActions');
  const yearEl = document.getElementById('year');
  const primaryCta = document.getElementById('primaryCta');

  function updateHeaderScrolled() {
    if (!siteHeader) return;
    siteHeader.classList.toggle('is-scrolled', window.scrollY > 8);
  }

  if (siteHeader) {
    updateHeaderScrolled();
    window.addEventListener('scroll', updateHeaderScrolled, { passive: true });
  }

  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  function isLoggedIn() {
    try {
      return Boolean(window.localStorage.getItem('mira_session'));
    } catch (_) {
      return false;
    }
  }

  function renderAuth() {
    if (!headerActions) return;
    const loggedIn = isLoggedIn();

    if (loggedIn) {
      headerActions.innerHTML = `
        <a href="/assistant" class="login-btn">Open assistant</a>
        <button type="button" class="signup-btn" id="logoutBtn">Logout</button>
      `;
      const logoutBtn = document.getElementById('logoutBtn');
      if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
          try {
            window.localStorage.removeItem('mira_session');
          } catch (_) {}
          renderAuth();
        });
      }
      if (primaryCta) primaryCta.href = '/assistant';
    } else {
      headerActions.innerHTML = `
        <a href="login.html" class="login-btn" id="loginLink">Login</a>
        <a href="signup.html" class="signup-btn" id="signupLink">Sign up</a>
      `;
      if (primaryCta) primaryCta.href = 'login.html';
    }
  }

  if (headerActions && headerActions.getAttribute('data-auth') !== 'false') {
    renderAuth();
  }
})();
