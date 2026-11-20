'use strict';

function setupCover() {
  const cover = document.querySelector('#top');
  const video = document.querySelector('#cover-video');
  const button = document.querySelector('#cover-toggle');
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const compact = window.matchMedia('(max-width: 640px)');
  let motion = !preference.matches && !navigator.connection?.saveData;
  let visible = true;

  function updateLabel() {
    const label = motion ? 'Pause cover video' : 'Play cover video';
    button.setAttribute('aria-label', label);
    button.title = label;
    button.firstElementChild.className = motion ? 'pause-glyph' : 'play-glyph';
  }

  function update() {
    updateLabel();
    const source = compact.matches ? video.dataset.mobileSrc : video.dataset.src;
    if (video.hasAttribute('src') && video.getAttribute('src') !== source) {
      video.classList.remove('ready');
      video.removeAttribute('src');
      video.load();
    }
    if (!motion || !visible || document.hidden) {
      video.pause();
      return;
    }
    if (!video.hasAttribute('src')) video.src = source;
    video.play().catch(error => {
      if (error.name === 'AbortError') return;
      motion = false;
      updateLabel();
    });
  }

  button.hidden = false;
  video.addEventListener('playing', () => video.classList.add('ready'));
  compact.addEventListener('change', update);
  button.addEventListener('click', () => {
    motion = !motion;
    update();
  });
  preference.addEventListener('change', () => {
    motion = !preference.matches && !navigator.connection?.saveData;
    update();
  });
  document.addEventListener('visibilitychange', update);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    update();
  }).observe(cover);
  update();
}

function waitForMedia(video, event, signal) {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(event, ready);
      video.removeEventListener('error', failed);
      signal.removeEventListener('abort', aborted);
    };
    function ready() { cleanup(); resolve(); }
    function failed() { cleanup(); reject(new Error('Unable to load this clip')); }
    function aborted() { cleanup(); reject(signal.reason); }
    const timer = setTimeout(failed, 25000);
    video.addEventListener(event, ready);
    video.addEventListener('error', failed);
    signal.addEventListener('abort', aborted, {once: true});
  });
}

async function prepareVideo(video, signal) {
  if (video.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && !video.error) return;
  const ready = waitForMedia(video, 'canplaythrough', signal);
  video.preload = 'auto';
  if (video.error || video.networkState !== HTMLMediaElement.NETWORK_LOADING) video.load();
  await ready;
}

async function alignVideo(video, time, signal) {
  signal.throwIfAborted();
  const target = Math.min(time, Math.max(0, video.duration - .02));
  if (Math.abs(video.currentTime - target) > .015 || video.seeking) {
    const seeked = waitForMedia(video, 'seeked', signal);
    video.currentTime = target;
    await seeked;
  }
}

function keepInSync(members, signal) {
  const [reference, ...others] = members;
  function tick() {
    if (signal.aborted) return;
    if (!reference.seeking && !reference.ended) {
      const time = reference.currentTime;
      for (const video of others) {
        if (!video.paused && !video.seeking && Math.abs(video.currentTime - time) > .075) {
          video.currentTime = Math.min(time, video.duration);
        }
      }
    }
    requestAnimationFrame(tick);
  }
  tick();
}

function setupGallery() {
  const cases = JSON.parse(document.querySelector('#comparison-data').textContent);
  const model = document.querySelector('#model-select');
  const example = document.querySelector('#example-select');
  const grid = document.querySelector('#clip-grid');
  const film = document.querySelector('#project-film');
  const status = document.querySelector('#play-status');
  const play = document.querySelector('#group-play');
  const replay = document.querySelector('#group-replay');
  const prompt = document.querySelector('#case-prompt');
  const context = document.querySelector('#case-context');
  let phase = 'idle';
  let operation;
  let videos = Array.from(grid.querySelectorAll('video'));

  function playbackState(next, message) {
    phase = next;
    const running = phase !== 'idle';
    play.querySelector('.action-text').textContent = running ? 'Pause together' : 'Play together';
    play.firstElementChild.className = running ? 'pause-glyph' : 'play-glyph';
    status.textContent = message;
  }

  function stop(message = '') {
    operation?.abort();
    operation = undefined;
    playbackState('idle', message);
    videos.forEach(video => video.pause());
  }

  async function start(rewind) {
    const first = videos[0];
    const time = rewind || first.ended ? 0 : first.currentTime;
    stop();
    const job = new AbortController();
    operation = job;
    const members = videos.slice();
    playbackState('loading', 'Loading the comparison…');
    try {
      // First playback initializes each decoder. Hold each clip as it becomes ready
      // so a slow download or decoder cannot give its neighbors a head start.
      await Promise.all(members.map(async video => {
        await prepareVideo(video, job.signal);
        job.signal.throwIfAborted();
        await video.play();
        if (!job.signal.aborted) video.pause();
      }));
      await Promise.all(members.map(video => alignVideo(video, time, job.signal)));
      job.signal.throwIfAborted();
      await Promise.all(members.map(video => video.play()));
      if (!job.signal.aborted) {
        keepInSync(members, job.signal);
        playbackState('playing', 'Playing together');
      }
    } catch {
      if (!job.signal.aborted) stop('A clip could not play. Retry or use its individual controls.');
    }
  }

  function fillExamples() {
    const choices = cases.filter(item => item.model === model.value);
    example.replaceChildren(...choices.map(item => new Option(item.title + ' · ' + item.target, item.id)));
  }

  function selectCase(item) {
    stop();
    for (const video of videos) {
      video.removeAttribute('src');
      video.load();
    }
    const type = item.figure === 4 ? 'Main comparison'
      : item.figure === 5 ? 'Scheduler variants'
      : item.figure === 8 ? 'Matched reuse budget' : 'Additional comparison';
    context.textContent = [item.model, item.target + ' target', type].join(' · ');
    prompt.textContent = '“' + item.prompt + '”';
    const clips = item.clips.map(clip => {
      const figure = document.createElement('figure');
      figure.className = 'clip';
      figure.classList.toggle('ours', clip.method.endsWith('MORCA'));
      const video = document.createElement('video');
      video.controls = true;
      video.muted = true;
      video.playsInline = true;
      video.preload = 'none';
      video.width = item.width;
      video.height = item.height;
      video.poster = clip.path.replace(/\.mp4$/, '.webp');
      video.src = clip.path;
      video.setAttribute('aria-label', clip.method + ', ' + item.title);
      const caption = document.createElement('figcaption');
      caption.textContent = clip.method;
      if (clip.method === 'Original') {
        const label = document.createElement('small');
        label.textContent = 'No cache';
        caption.append(label);
      }
      figure.append(video, caption);
      return figure;
    });
    grid.replaceChildren(...clips);
    videos = Array.from(grid.querySelectorAll('video'));
  }

  document.querySelector('#watch-film').addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    film.play().catch(() => film.focus({ preventScroll: true }));
  });
  film.addEventListener('play', () => stop());
  grid.addEventListener('play', () => film.pause(), true);
  play.addEventListener('click', () => phase === 'idle' ? start(false) : stop('Paused'));
  replay.addEventListener('click', () => start(true));
  grid.addEventListener('pause', event => {
    if (phase === 'playing' && event.target.paused && !event.target.ended) stop('Paused');
  }, true);
  grid.addEventListener('ended', () => {
    if (phase === 'playing' && videos.every(video => video.ended)) stop('Finished');
  }, true);
  model.addEventListener('change', () => {
    fillExamples();
    selectCase(cases.find(item => item.id === example.value));
  });
  example.addEventListener('change', () => selectCase(cases.find(item => item.id === example.value)));
  document.querySelectorAll('a[data-example]').forEach(link => {
    link.addEventListener('click', () => {
      const item = cases.find(item => item.id === link.dataset.example);
      model.value = item.model;
      fillExamples();
      example.value = item.id;
      selectCase(item);
    });
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && phase !== 'idle') stop('Paused');
  });
  model.value = cases[0].model;
  fillExamples();
  document.querySelector('#gallery-tools').hidden = false;
}

setupGallery();
setupCover();
