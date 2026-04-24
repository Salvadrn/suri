import dotenv from 'dotenv';
dotenv.config({ override: true });
import express from 'express';
import multer from 'multer';
import Anthropic from '@anthropic-ai/sdk';
import { createClient as createDeepgram } from '@deepgram/sdk';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

const execFileAsync = promisify(execFile);

const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.static('.', { dotfiles: 'deny', index: 'index.html' }));

const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 2 * 1024 * 1024 * 1024 },
});

const deepgram = createDeepgram(process.env.DEEPGRAM_API_KEY);
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const MODEL = 'claude-haiku-4-5-20251001';

function tmpFile(ext) {
  return path.join(os.tmpdir(), `s2t-${randomUUID()}.${ext}`);
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

async function extractAudioFromFile(filePath) {
  const out = tmpFile('mp3');
  await execFileAsync('ffmpeg', ['-i', filePath, '-vn', '-acodec', 'libmp3lame', '-ab', '128k', '-y', out]);
  return out;
}

async function downloadYouTubeAudio(url) {
  try { new URL(url); } catch { throw new Error('Invalid URL'); }
  const out = tmpFile('mp3');
  await execFileAsync('yt-dlp', ['-x', '--audio-format', 'mp3', '--audio-quality', '5', '-o', out, url]);
  return out;
}

async function transcribeAudio(audioPath) {
  const audioBuffer = fs.readFileSync(audioPath);
  const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
    audioBuffer,
    {
      model: 'nova-2',
      smart_format: true,
      punctuate: true,
      diarize: true,
      utterances: true,
      detect_language: true,
    }
  );
  if (error) throw new Error(`Deepgram error: ${error.message || error}`);
  return result;
}

async function enrichSubtitles(utterances) {
  if (utterances.length === 0) return [];
  const lines = utterances.map((u, i) => `${i}: ${u.transcript}`).join('\n');
  const msg = await anthropic.messages.create({
    model: MODEL,
    max_tokens: 4000,
    system: 'You analyze educational video transcripts and infer plausible emotion and background sound for each line. Output valid JSON only.',
    messages: [
      {
        role: 'user',
        content: `For each numbered line, return a JSON array with objects { "i": number, "emotion": string, "sound": string }.

Emotion must be one of: neutral, happy, sad, excited, calm, curious, angry.
Sound is a short description of plausible background ambience (e.g. "soft music", "classroom ambience", "silence", "applause"). Keep under 8 words.

Lines:
${lines}

Return ONLY the JSON array, no markdown, no commentary.`,
      },
      { role: 'assistant', content: '[' },
    ],
  });
  const raw = '[' + msg.content[0].text;
  const jsonEnd = raw.lastIndexOf(']');
  return JSON.parse(raw.slice(0, jsonEnd + 1));
}

function buildSubtitles(utterances, enriched) {
  const enrichMap = new Map(enriched.map((e) => [e.i, e]));
  return utterances.map((u, i) => {
    const speakerLabel =
      u.speaker !== undefined && u.speaker !== null
        ? `Speaker ${u.speaker + 1}`
        : 'Narrator';
    const e = enrichMap.get(i) || {};
    return {
      time: formatTime(u.start),
      end: formatTime(u.end),
      speaker: speakerLabel,
      text: u.transcript,
      emotion: e.emotion || 'neutral',
      sound: e.sound || 'Silence',
    };
  });
}

async function processAudio(audioPath) {
  const result = await transcribeAudio(audioPath);
  const utterances = result.results?.utterances || [];
  const enriched = await enrichSubtitles(utterances);
  return {
    subtitles: buildSubtitles(utterances, enriched),
    language: result.results?.channels?.[0]?.detected_language || 'en',
  };
}

function safeUnlink(p) {
  try { fs.unlinkSync(p); } catch {}
}

app.post('/api/transcribe-file', upload.single('video'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

  let audioPath;
  try {
    audioPath = await extractAudioFromFile(req.file.path);
    const data = await processAudio(audioPath);
    res.json(data);
  } catch (err) {
    console.error('transcribe-file failed:', err);
    res.status(500).json({ error: err.message });
  } finally {
    safeUnlink(req.file.path);
    if (audioPath) safeUnlink(audioPath);
  }
});

app.post('/api/transcribe-url', async (req, res) => {
  const { url } = req.body || {};
  if (!url) return res.status(400).json({ error: 'No URL provided' });

  let audioPath;
  try {
    audioPath = await downloadYouTubeAudio(url);
    const data = await processAudio(audioPath);
    res.json(data);
  } catch (err) {
    console.error('transcribe-url failed:', err);
    res.status(500).json({ error: err.message });
  } finally {
    if (audioPath) safeUnlink(audioPath);
  }
});

app.post('/api/translate', async (req, res) => {
  const { subtitles, language } = req.body || {};
  if (!Array.isArray(subtitles) || !language) {
    return res.status(400).json({ error: 'Missing subtitles or language' });
  }

  try {
    const lines = subtitles.map((s, i) => `${i}: ${s.text}`).join('\n');
    const langNames = {
      EN: 'English', ES: 'Spanish', FR: 'French', DE: 'German',
      PT: 'Portuguese', ZH: 'Chinese (Simplified)', AR: 'Arabic',
    };
    const targetName = langNames[language] || language;

    const msg = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 4000,
      system: `Translate subtitle lines to ${targetName}. Preserve numbering. Output valid JSON only.`,
      messages: [
        {
          role: 'user',
          content: `Translate each line below to ${targetName}.
Return a JSON array of objects { "i": number, "text": string }.

${lines}

Return ONLY the JSON array.`,
        },
        { role: 'assistant', content: '[' },
      ],
    });

    const raw = '[' + msg.content[0].text;
    const jsonEnd = raw.lastIndexOf(']');
    const translations = JSON.parse(raw.slice(0, jsonEnd + 1));
    const tMap = new Map(translations.map((t) => [t.i, t.text]));
    const translated = subtitles.map((s, i) => ({ ...s, text: tMap.get(i) ?? s.text }));
    res.json({ subtitles: translated });
  } catch (err) {
    console.error('translate failed:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    anthropic: !!process.env.ANTHROPIC_API_KEY,
    deepgram: !!process.env.DEEPGRAM_API_KEY,
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  if (!process.env.ANTHROPIC_API_KEY) console.warn('Missing ANTHROPIC_API_KEY');
  if (!process.env.DEEPGRAM_API_KEY) console.warn('Missing DEEPGRAM_API_KEY');
});
