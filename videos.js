'use strict';

// Bring moving content into view before starting its decoders. A tap or Space
// pauses the film or the entire comparison without putting controls over frames.
function followViewport(surface, setActive, isPaused, threshold = 0) {
  const preference = matchMedia('(prefers-reduced-motion: reduce)');
  let visible = false;
  let enabled = !preference.matches;
  let active = false;

  function update(force = false) {
    const next = visible && enabled && !document.hidden;
    if (next !== active || force) {
      active = next;
      setActive(active);
    }
  }

  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting && entry.intersectionRatio >= threshold;
    update();
  }, {threshold}).observe(surface);
  document.addEventListener('visibilitychange', () => update());
  preference.addEventListener('change', () => {
    enabled = !preference.matches;
    update();
  });
  if (isPaused) {
    const toggle = () => {
      enabled = isPaused();
      update(true);
    };
    surface.addEventListener('click', toggle);
    surface.addEventListener('keydown', event => {
      if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        if (!event.repeat) toggle();
      }
    });
  }
}

function setupCover() {
  const video = document.querySelector('#cover-video');
  const compact = matchMedia('(max-width: 640px)');
  let active = false;
  function update() {
    const source = compact.matches ? video.dataset.mobileSrc : video.dataset.src;
    if (source !== video.getAttribute('src')) {
      video.classList.remove('ready');
      if (active) video.src = source;
    }
    if (active) video.play().catch(() => {}); // The poster remains visible when autoplay is blocked.
    else video.pause();
  }
  video.addEventListener('playing', () => video.classList.add('ready'));
  compact.addEventListener('change', update);
  followViewport(video.parentElement, next => {
    active = next && !navigator.connection?.saveData;
    update();
  });
}

function setupFilm() {
  const film = document.querySelector('#project-film');
  const player = film.closest('.film-player');
  const sound = document.querySelector('#film-sound');
  const captions = document.querySelector('#film-captions');
  const position = document.querySelector('#film-position');
  const fullscreen = document.querySelector('#film-fullscreen');
  const track = film.querySelector('track').track;
  track.mode = 'showing';

  followViewport(film, active => {
    if (active) film.play().catch(() => {}); // A tap can retry under a device's autoplay policy.
    else film.pause();
  }, () => film.paused, .25);
  sound.addEventListener('click', () => { film.muted = !film.muted; });
  film.addEventListener('volumechange', () => {
    sound.textContent = film.muted ? 'Sound off' : 'Sound on';
    sound.setAttribute('aria-pressed', String(!film.muted));
  });
  captions.addEventListener('click', () => {
    track.mode = track.mode === 'showing' ? 'disabled' : 'showing';
    captions.setAttribute('aria-pressed', String(track.mode === 'showing'));
  });
  film.addEventListener('loadedmetadata', () => { position.max = film.duration; });
  film.addEventListener('timeupdate', () => { position.value = film.currentTime; });
  position.addEventListener('input', () => {
    if (Number.isFinite(film.duration)) film.currentTime = position.valueAsNumber;
  });
  if (player.requestFullscreen || film.webkitEnterFullscreen) {
    fullscreen.hidden = false;
    fullscreen.addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      else if (player.requestFullscreen) player.requestFullscreen().catch(() => {});
      else film.webkitEnterFullscreen();
    });
    document.addEventListener('fullscreenchange', () => {
      fullscreen.textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen';
    });
  }
  player.querySelector('.film-tools').hidden = false;
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
  const status = document.querySelector('#play-status');
  const prompt = document.querySelector('#case-prompt');
  const context = document.querySelector('#case-context');
  let active = false;
  let operation;
  let videos = Array.from(grid.querySelectorAll('video'));

  function stop() {
    operation?.abort();
    operation = undefined;
    status.textContent = '';
    videos.forEach(video => video.pause());
  }

  async function start() {
    if (operation) return;
    const time = videos[0].currentTime;
    const job = new AbortController();
    operation = job;
    const members = videos.slice();
    try {
      // Prewarm every decoder before aligning the group to one source timestamp.
      await Promise.all(members.map(async video => {
        await prepareVideo(video, job.signal);
        job.signal.throwIfAborted();
        await video.play();
        if (!job.signal.aborted) video.pause();
      }));
      await Promise.all(members.map(video => alignVideo(video, time, job.signal)));
      job.signal.throwIfAborted();
      await Promise.all(members.map(video => video.play()));
      keepInSync(members, job.signal);
    } catch {
      if (!job.signal.aborted) {
        stop();
        status.textContent = 'Tap a clip to start this comparison.';
      }
    }
  }

  function fillExamples() {
    const choices = cases.filter(item => item.model === model.value);
    example.replaceChildren(...choices.map(item => new Option(item.title + ' · ' + item.target, item.id)));
  }

  function selectCase() {
    const item = cases.find(candidate => candidate.id === example.value);
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
      video.defaultMuted = true;
      video.muted = true;
      video.playsInline = true;
      video.loop = true;
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
    if (active) start();
  }

  followViewport(grid, next => {
    active = next;
    if (active) start();
    else stop();
  }, () => !operation);
  model.addEventListener('change', () => {
    fillExamples();
    selectCase();
  });
  example.addEventListener('change', selectCase);
  model.value = cases[0].model;
  fillExamples();
  document.querySelector('#gallery-tools').hidden = false;
}

setupCover();
setupFilm();
setupGallery();
