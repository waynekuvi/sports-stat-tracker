async function fetchStats() {
  try {
    const res = await fetch('/api/stats', { cache: 'no-store' });
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();

    document.getElementById('mode').textContent = data.mode;
    document.getElementById('homeGoals').textContent = data.homeGoals;
    document.getElementById('awayGoals').textContent = data.awayGoals;
    document.getElementById('homePossession').textContent = data.homePossession;
    document.getElementById('awayPossession').textContent = data.awayPossession;
    document.getElementById('prob').textContent = data.probHomeNextGoal.toFixed(3);

    // Cache-bust the plot image so it refreshes
    const plot = document.getElementById('plot');
    const url = new URL(plot.src, window.location.origin);
    url.searchParams.set('t', Date.now().toString());
    plot.src = url.toString();
  } catch (e) {
    console.error('Failed to fetch stats', e);
  }
}

function start() {
  fetchStats();
  setInterval(fetchStats, 3000);
}

window.addEventListener('DOMContentLoaded', start);


