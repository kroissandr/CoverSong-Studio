// When the page is shown inside the app (a tab of the covers page, ?embed=1): no own header, and the frame's
// height follows the content so the page scrolls as one.
(function () {
  "use strict";
  const embedded = new URLSearchParams(location.search).get("embed") === "1" && window.parent !== window;
  if (!embedded) return;
  document.documentElement.classList.add("embed");
  let last = 0;
  function post() {
    const h = Math.ceil(document.body.getBoundingClientRect().height) + 24;
    if (Math.abs(h - last) > 2) { last = h; window.parent.postMessage({ scHeight: h }, location.origin); }
  }
  window.addEventListener("load", post);
  document.addEventListener("DOMContentLoaded", () => {
    post();
    if (window.ResizeObserver) new ResizeObserver(post).observe(document.body);
    setInterval(post, 1000);
  });
})();
