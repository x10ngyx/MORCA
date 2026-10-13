document.querySelectorAll('.video-comparison').forEach(group => {
  const videos = [...group.querySelectorAll('video')];
  const buttons = [...group.querySelectorAll('button')];
  const status = group.querySelector('.video-status');
  let operation = 0;
  group.addEventListener('click', async event => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const current = ++operation;
    const action = button.dataset.action;
    videos.forEach(video => video.pause());
    if (action === 'pause') {
      status.textContent = 'Paused';
      return;
    }
    buttons.filter(b => b.dataset.action !== 'pause').forEach(b => b.disabled = true);
    status.textContent = 'Loading videos…';
    const time = action === 'restart' || videos[0].ended ? 0 : videos[0].currentTime;
    const results = await Promise.allSettled(videos.map(async video => {
      if (video.readyState < 2) {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => finish(new Error('Video loading timed out')), 20000);
          const ready = () => finish();
          const error = () => finish(new Error('Video unavailable'));
          function finish(err) {
            clearTimeout(timer);
            video.removeEventListener('loadeddata', ready);
            video.removeEventListener('error', error);
            err ? reject(err) : resolve();
          }
          video.addEventListener('loadeddata', ready, {once: true});
          video.addEventListener('error', error, {once: true});
          video.load();
        });
      }
      video.currentTime = time;
    }));
    if (current === operation) {
      const playback = await Promise.allSettled(videos.map((video, i) =>
        results[i].status === 'fulfilled' ? video.play() : Promise.reject(results[i].reason)));
      if (current === operation) status.textContent = playback.some(r => r.status === 'rejected')
        ? 'Some videos could not play. Try their individual controls or replay.' : 'Playing';
      else videos.forEach(video => video.pause());
    }
    buttons.forEach(b => b.disabled = false);
  });
  videos.forEach(video => video.addEventListener('ended', () => {
    if (videos.every(v => v.ended)) status.textContent = 'Finished — replay to compare again';
  }));
});
