/* Live updates: polls /api/live every 5 s.
 *  - shows a pop-up + beep when a NEW fraud alert is raised (option 4)
 *  - keeps the red badge on the "Alerts" menu up to date
 *  - on Dashboard / Alerts / Transactions pages, reloads when new data arrives (option 2) */
(function () {
  var POLL_MS = 5000;
  var autoRefresh = document.body.dataset.autorefresh === "1";
  var last = null;

  function esc(t) { var d = document.createElement("div"); d.textContent = t; return d.innerHTML; }

  function showToast(a) {
    var box = document.getElementById("toasts");
    if (!box || !a) return;
    var el = document.createElement("div");
    el.className = "toast";
    el.innerHTML = "<strong>&#9888; New fraud alert</strong><br>Account " + esc(a.account_id) +
      " &middot; KES " + Number(a.amount).toLocaleString() + " &middot; " + esc(a.risk_level) +
      " risk<br><a href='/alerts/" + a.id + "'>Review now</a>";
    box.appendChild(el);
    setTimeout(function () { el.remove(); }, 10000);
  }

  function beep() {
    try {
      var ctx = new (window.AudioContext || window.webkitAudioContext)();
      var o = ctx.createOscillator(), g = ctx.createGain();
      o.connect(g); g.connect(ctx.destination);
      o.frequency.value = 880; g.gain.value = 0.08; o.start();
      setTimeout(function () { o.stop(); ctx.close(); }, 250);
    } catch (e) { /* browsers may block sound until the page has been clicked */ }
  }

  function isTyping() {
    var a = document.activeElement;
    return a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT");
  }

  // A reload would erase the pop-up, so re-show it right after the page reloads.
  try {
    var pending = sessionStorage.getItem("pendingToast");
    if (pending) { sessionStorage.removeItem("pendingToast"); showToast(JSON.parse(pending)); }
  } catch (e) {}

  function poll() {
    fetch("/api/live", { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return;
        var badge = document.getElementById("alert-badge");
        if (badge) { badge.textContent = d.open_alerts; badge.hidden = d.open_alerts === 0; }
        var newAlert = last && d.latest_alert_id > last.latest_alert_id;
        var changed = last && (d.total !== last.total || d.open_alerts !== last.open_alerts);
        if (newAlert) {
          beep();
          if (autoRefresh && !isTyping()) {
            try { sessionStorage.setItem("pendingToast", JSON.stringify(d.latest_alert)); } catch (e) {}
          } else { showToast(d.latest_alert); }
        }
        last = d;
        if (autoRefresh && changed && !isTyping()) setTimeout(function () { location.reload(); }, 400);
      })
      .catch(function () { /* server restarting or offline: try again next time */ });
  }
  poll();
  setInterval(poll, POLL_MS);
})();
