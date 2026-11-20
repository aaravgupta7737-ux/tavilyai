import 'dotenv/config';
import crypto from 'crypto';
import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel, Type, type Part } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const IS_PROD = process.env.NODE_ENV === 'production';

const COOKIE_NAME = 'tavily_auth_session';
const SESSION_SECRET =
  process.env.SESSION_SECRET ||
  crypto.createHash('sha256').update(process.env.GEMINI_API_KEY || 'tavily-local-dev-secret').digest('hex');

interface AuthUser {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  provider: 'google' | 'phone';
}

interface AttachmentPayload {
  id: string;
  name: string;
  mimeType: string;
  kind: 'image' | 'document';
  size: number;
  dataUrl?: string;
  textContent?: string;
}

interface PendingOtp {
  code: string;
  name: string;
  expiresAt: number;
}

const pendingOtps = new Map<string, PendingOtp>();
const DATA_DIR = path.resolve(__dirname, '.data');
const SESSIONS_FILE = path.join(DATA_DIR, 'sessions.json');
const userSessionsStore = new Map<string, unknown[]>();

function loadPersistedSessions() {
  try {
    if (!fs.existsSync(SESSIONS_FILE)) return;
    const raw = fs.readFileSync(SESSIONS_FILE, 'utf8');
    const parsed = JSON.parse(raw) as Record<string, unknown[]>;
    if (parsed && typeof parsed === 'object') {
      for (const [uid, list] of Object.entries(parsed)) {
        if (Array.isArray(list)) {
          userSessionsStore.set(uid, list);
        }
      }
    }
  } catch {
    // Ignore corrupt cache file
  }
}

function savePersistedSessions() {
  try {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }
    const obj: Record<string, unknown[]> = {};
    for (const [uid, list] of userSessionsStore.entries()) {
      obj[uid] = list;
    }
    fs.writeFileSync(SESSIONS_FILE, JSON.stringify(obj), 'utf8');
  } catch {
    // Ignore write failure
  }
}

loadPersistedSessions();

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured. Add it in Settings > Secrets.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function parseCookies(cookieHeader: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!cookieHeader) return out;
  for (const part of cookieHeader.split(';')) {
    const idx = part.indexOf('=');
    if (idx === -1) continue;
    const key = part.slice(0, idx).trim();
    const val = decodeURIComponent(part.slice(idx + 1).trim());
    out[key] = val;
  }
  return out;
}

function signUserCookie(user: AuthUser): string {
  const payload = Buffer.from(JSON.stringify(user), 'utf8').toString('base64url');
  const sig = crypto.createHmac('sha256', SESSION_SECRET).update(payload).digest('base64url');
  return `${payload}.${sig}`;
}

function verifyUserCookie(token: string | undefined): AuthUser | null {
  if (!token) return null;
  const parts = token.split('.');
  if (parts.length !== 2) return null;
  const [payload, sig] = parts;
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(payload)
    .digest('base64url');
  const sigBuf = Buffer.from(sig);
  const expBuf = Buffer.from(expectedSig);
  if (sigBuf.length !== expBuf.length || !crypto.timingSafeEqual(sigBuf, expBuf)) {
    return null;
  }
  try {
    return JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as AuthUser;
  } catch {
    return null;
  }
}

function setAuthCookie(res: express.Response, user: AuthUser) {
  const token = signUserCookie(user);
  res.cookie(COOKIE_NAME, token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 30 * 24 * 60 * 60 * 1000,
    path: '/',
  });
}

function extractDomain(urlStr: string, fallbackTitle?: string): string {
  try {
    const parsed = new URL(urlStr);
    const host = parsed.hostname.replace(/^www\./, '').toLowerCase();
    if (
      host === 'vertexaisearch.cloud.google.com' &&
      fallbackTitle &&
      /^[a-z0-9.-]+\.[a-z]{2,}$/i.test(fallbackTitle.trim())
    ) {
      return fallbackTitle.trim().replace(/^www\./, '').toLowerCase();
    }
    return host;
  } catch {
    return urlStr.toLowerCase();
  }
}

function maskPhone(phone: string): string {
  const digits = phone.replace(/\D/g, '');
  if (digits.length < 4) return phone;
  return `${phone.slice(0, Math.max(0, phone.length - 4)).replace(/\d/g, '•')}${phone.slice(-4)}`;
}

function buildAttachmentParts(attachments: AttachmentPayload[] = []): Part[] {
  const parts: Part[] = [];
  for (const att of attachments.slice(0, 6)) {
    if (att.dataUrl && att.dataUrl.includes(',')) {
      const [meta, base64] = att.dataUrl.split(',', 2);
      const mimeMatch = meta.match(/data:([^;]+);base64/);
      const mimeType = mimeMatch?.[1] || att.mimeType || 'application/octet-stream';
      if (base64) {
        parts.push({
          inlineData: {
            mimeType,
            data: base64,
          },
        });
      }
    } else if (att.textContent) {
      parts.push({
        text: `[Attached Document: ${att.name}]\n${att.textContent.slice(0, 20000)}`,
      });
    }
  }
  return parts;
}

function getTimeRangeInstruction(timeRange: string): string {
  switch (timeRange) {
    case 'day':
      return 'Prioritize results and developments from the past 24 hours.';
    case 'week':
      return 'Prioritize results and developments from the past 7 days.';
    case 'month':
      return 'Prioritize results and developments from the past 30 days.';
    case 'year':
      return 'Prioritize results and developments from the past 12 months.';
    default:
      return '';
  }
}

function getFormatInstruction(format: string): string {
  switch (format) {
    case 'comparison_matrix':
      return 'Emphasize structured side-by-side comparison tables, metrics, pros/cons, and explicit trade-offs across options.';
    case 'technical_deep_dive':
      return 'Provide rigorous technical depth covering architecture, specifications, benchmarks, implementation constraints, and failure modes.';
    case 'action_plan':
      return 'Prioritize concrete execution steps, sequencing, risk mitigation, ownership, and measurable next actions.';
    default:
      return 'Provide a crisp, executive-ready analytical brief with clear evidence, quantitative metrics, and practical takeaways.';
  }
}

async function withRetry<T>(fn: () => Promise<T>, retries = 1): Promise<T> {
  try {
    return await fn();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    const retryable =
      msg.includes('503') ||
      msg.includes('500') ||
      msg.includes('UNAVAILABLE');
    if (retries > 0 && retryable) {
      await new Promise((resolve) => setTimeout(resolve, 900));
      return withRetry(fn, retries - 1);
    }
    throw err;
  }
}

interface GroundingChunk {
  web?: {
    uri?: string;
    title?: string;
  };
}

interface GroundingSupport {
  segment?: {
    text?: string;
  };
  groundingChunkIndices?: number[];
}

interface WebSearchHit {
  title: string;
  url: string;
  snippet: string;
  publishedDate?: string;
}

const ALLOWED_MODELS = new Set(['gemini-4-hydrogen', 'gemini-3.1-flash-lite']);
function resolveModelName(model?: string): string {
  return model && ALLOWED_MODELS.has(model) ? model : 'gemini-4-hydrogen';
}

let hydrogenCooldownUntil = 0;
let googleSearchCooldownUntil = 0;

function isQuotaOrUnavailableError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    msg.includes('429') ||
    msg.includes('RESOURCE_EXHAUSTED') ||
    msg.includes('503') ||
    msg.includes('UNAVAILABLE')
  );
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out (${ms}ms)`)), ms);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function generateWithModelFallback(
  ai: GoogleGenAI,
  preferredModel: string,
  buildRequest: (modelName: string) => Parameters<typeof ai.models.generateContent>[0],
  options?: { maxAttempts?: number; timeoutMs?: number }
) {
  const effectiveModel =
    preferredModel === 'gemini-4-hydrogen' && Date.now() < hydrogenCooldownUntil
      ? 'gemini-3.1-flash-lite'
      : preferredModel;

  const fallbackModel =
    effectiveModel === 'gemini-3.1-flash-lite'
      ? 'gemini-4-hydrogen'
      : 'gemini-3.1-flash-lite';

  const baseTimeout = options?.timeoutMs ?? 20000;
  const maxAttempts = options?.maxAttempts ?? 4;

  const allCandidates = [
    { model: effectiveModel, timeoutMs: baseTimeout, delayMs: 0 },
    { model: fallbackModel, timeoutMs: baseTimeout, delayMs: 400 },
    { model: effectiveModel, timeoutMs: baseTimeout + 4000, delayMs: 1200 },
    { model: fallbackModel, timeoutMs: baseTimeout + 4000, delayMs: 1800 },
  ].slice(0, maxAttempts);

  let lastErr: unknown;
  for (const attempt of allCandidates) {
    if (attempt.delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, attempt.delayMs));
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), attempt.timeoutMs);
    try {
      const req = buildRequest(attempt.model);
      req.config = {
        ...(req.config || {}),
        abortSignal: controller.signal,
      };
      return await withTimeout(
        ai.models.generateContent(req),
        attempt.timeoutMs + 500,
        attempt.model
      );
    } catch (err) {
      lastErr = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isAbortOrTimeout =
        controller.signal.aborted ||
        msg.includes('timed out') ||
        msg.includes('AbortError') ||
        msg.includes('aborted');
      if (attempt.model === 'gemini-4-hydrogen' && (isQuotaOrUnavailableError(err) || isAbortOrTimeout)) {
        hydrogenCooldownUntil = Date.now() + 120_000;
      }
      if (!isQuotaOrUnavailableError(err) && !isAbortOrTimeout && !msg.includes('500')) {
        throw err;
      }
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastErr;
}

function decodeXmlEntities(str: string): string {
  return str
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function fetchTavilyResults(
  query: string,
  depth: string,
  timeRange: string,
  includeDomains: string[],
  excludeDomains: string[]
): Promise<{ hits: WebSearchHit[]; answer?: string } | null> {
  const apiKey = process.env.TAVILY_API_KEY;
  if (!apiKey) return null;
  try {
    const resp = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        search_depth: depth === 'deep' ? 'advanced' : 'basic',
        include_answer: true,
        max_results: 6,
        ...(timeRange !== 'any' ? { time_range: timeRange } : {}),
        ...(includeDomains.length > 0 ? { include_domains: includeDomains } : {}),
        ...(excludeDomains.length > 0 ? { exclude_domains: excludeDomains } : {}),
      }),
      signal: AbortSignal.timeout(12000),
    });
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      answer?: string;
      results?: Array<{ title?: string; url?: string; content?: string; published_date?: string }>;
    };
    const hits: WebSearchHit[] = (data.results || [])
      .filter((r) => Boolean(r.url))
      .map((r) => ({
        title: r.title?.trim() || extractDomain(r.url!),
        url: r.url!.trim(),
        snippet: (r.content || '').trim().slice(0, 400),
        publishedDate: r.published_date,
      }));
    return { hits, answer: data.answer };
  } catch {
    return null;
  }
}

async function fetchLiveWebResults(
  query: string,
  includeDomains: string[] = [],
  excludeDomains: string[] = []
): Promise<WebSearchHit[]> {
  const siteQuery =
    includeDomains.length > 0
      ? `${query} (${includeDomains.slice(0, 3).map((d) => `site:${d}`).join(' OR ')})`
      : query;

  const isAllowedUrl = (url: string) => {
    const dom = extractDomain(url);
    return !excludeDomains.some((ex) => dom === ex || dom.endsWith(`.${ex}`));
  };

  const results: WebSearchHit[] = [];

  await Promise.allSettled([
    // 1. Wikipedia Search & Extracts
    (async () => {
      const wikiUrl = `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${encodeURIComponent(
        query
      )}&gsrlimit=4&prop=extracts|info&inprop=url&exintro=1&explaintext=1&exchars=750&format=json`;
      const r = await fetch(wikiUrl, {
        headers: { 'User-Agent': 'TavilyResearchAgent/1.0' },
        signal: AbortSignal.timeout(6000),
      });
      if (!r.ok) return;
      const json = (await r.json()) as {
        query?: {
          pages?: Record<
            string,
            { title?: string; fullurl?: string; extract?: string; index?: number }
          >;
        };
      };
      const pages = Object.values(json.query?.pages || {}).sort(
        (a, b) => (a.index ?? 99) - (b.index ?? 99)
      );
      for (const p of pages) {
        if (p.fullurl && p.extract && isAllowedUrl(p.fullurl)) {
          results.push({
            title: `${p.title} - Wikipedia`,
            url: p.fullurl,
            snippet: p.extract.trim(),
          });
        }
      }
    })(),

    // 2. Google News RSS
    (async () => {
      const rssUrl = `https://news.google.com/rss/search?q=${encodeURIComponent(
        siteQuery
      )}&hl=en-US&gl=US&ceid=US:en`;
      const r = await fetch(rssUrl, {
        headers: { 'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)' },
        signal: AbortSignal.timeout(6000),
      });
      if (!r.ok) return;
      const xml = await r.text();
      const itemMatches = xml.matchAll(/<item>([\s\S]*?)<\/item>/g);
      let count = 0;
      for (const m of itemMatches) {
        if (count >= 4) break;
        const block = m[1];
        const title = decodeXmlEntities(block.match(/<title>([\s\S]*?)<\/title>/)?.[1] || '');
        const link = decodeXmlEntities(block.match(/<link>([\s\S]*?)<\/link>/)?.[1] || '');
        const pubDate = decodeXmlEntities(
          block.match(/<pubDate>([\s\S]*?)<\/pubDate>/)?.[1] || ''
        );
        const sourceTitle = decodeXmlEntities(
          block.match(/<source[^>]*>([\s\S]*?)<\/source>/)?.[1] || ''
        );
        const desc = decodeXmlEntities(
          block.match(/<description>([\s\S]*?)<\/description>/)?.[1] || ''
        );
        if (title && link && isAllowedUrl(link)) {
          results.push({
            title,
            url: link,
            snippet: desc || (sourceTitle ? `Reported by ${sourceTitle}` : title),
            publishedDate: pubDate ? pubDate.slice(0, 16) : undefined,
          });
          count++;
        }
      }
    })(),

    // 3. Hacker News Algolia Search
    (async () => {
      const hnUrl = `https://hn.algolia.com/api/v1/search?query=${encodeURIComponent(
        query
      )}&tags=story&hitsPerPage=3`;
      const r = await fetch(hnUrl, { signal: AbortSignal.timeout(5000) });
      if (!r.ok) return;
      const json = (await r.json()) as {
        hits?: Array<{
          objectID: string;
          title?: string;
          url?: string;
          story_text?: string;
          created_at?: string;
          points?: number;
        }>;
      };
      for (const h of json.hits || []) {
        const targetUrl = h.url || `https://news.ycombinator.com/item?id=${h.objectID}`;
        if (h.title && isAllowedUrl(targetUrl)) {
          results.push({
            title: h.title,
            url: targetUrl,
            snippet: h.story_text
              ? decodeXmlEntities(h.story_text).slice(0, 350)
              : `${h.title} (${h.points ?? 0} points on Hacker News)`,
            publishedDate: h.created_at?.slice(0, 10),
          });
        }
      }
    })(),
  ]);

  if (includeDomains.length > 0) {
    const filtered = results.filter((r) => {
      const dom = extractDomain(r.url, r.title);
      return includeDomains.some((inc) => dom === inc || dom.endsWith(`.${inc}`));
    });
    if (filtered.length > 0) return filtered;
  }
  return results;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  // Auth Routes
  app.get('/api/auth/me', (req, res) => {
    const cookies = parseCookies(req.headers.cookie);
    const user = verifyUserCookie(cookies[COOKIE_NAME]);
    res.json({ user });
  });

  app.post('/api/auth/google', (req, res) => {
    const { name, email } = req.body as { name?: string; email?: string };
    if (!name?.trim() || !email?.trim()) {
      res.status(400).json({ error: 'Name and email are required.' });
      return;
    }
    const cleanEmail = email.trim().toLowerCase();
    const user: AuthUser = {
      id: `google_${crypto.createHash('sha1').update(cleanEmail).digest('hex').slice(0, 12)}`,
      name: name.trim(),
      email: cleanEmail,
      provider: 'google',
    };
    setAuthCookie(res, user);
    res.json({ user });
  });

  app.post('/api/auth/phone/send-otp', (req, res) => {
    const { name, phone } = req.body as { name?: string; phone?: string };
    if (!name?.trim() || !phone?.trim()) {
      res.status(400).json({ error: 'Name and phone number are required.' });
      return;
    }
    const cleanPhone = phone.trim();
    const code = String(Math.floor(100000 + Math.random() * 900000));
    pendingOtps.set(cleanPhone, {
      code,
      name: name.trim(),
      expiresAt: Date.now() + 10 * 60 * 1000,
    });
    res.json({
      ok: true,
      phone: cleanPhone,
      maskedPhone: maskPhone(cleanPhone),
      previewCode: code,
    });
  });

  app.post('/api/auth/phone/verify-otp', (req, res) => {
    const { phone, code, name } = req.body as {
      phone?: string;
      code?: string;
      name?: string;
    };
    if (!phone?.trim() || !code?.trim()) {
      res.status(400).json({ error: 'Phone number and verification code are required.' });
      return;
    }
    const pending = pendingOtps.get(phone.trim());
    if (!pending || pending.expiresAt < Date.now() || pending.code !== code.trim()) {
      res.status(400).json({ error: 'Invalid or expired verification code.' });
      return;
    }
    pendingOtps.delete(phone.trim());
    const user: AuthUser = {
      id: `phone_${crypto.createHash('sha1').update(phone.trim()).digest('hex').slice(0, 12)}`,
      name: name?.trim() || pending.name,
      phone: phone.trim(),
      provider: 'phone',
    };
    setAuthCookie(res, user);
    res.json({ user });
  });

  app.post('/api/auth/logout', (_req, res) => {
    res.clearCookie(COOKIE_NAME, { path: '/' });
    res.json({ ok: true });
  });

  // Session Storage Routes
  app.get('/api/sessions', (req, res) => {
    const cookies = parseCookies(req.headers.cookie);
    const user = verifyUserCookie(cookies[COOKIE_NAME]);
    if (!user) {
      res.json({ sessions: [] });
      return;
    }
    res.json({ sessions: userSessionsStore.get(user.id) || [] });
  });

  app.post('/api/sessions', (req, res) => {
    const cookies = parseCookies(req.headers.cookie);
    const user = verifyUserCookie(cookies[COOKIE_NAME]);
    if (!user) {
      res.status(401).json({ error: 'Not signed in.' });
      return;
    }
    const { session } = req.body as { session?: { id?: string } };
    if (!session?.id) {
      res.status(400).json({ error: 'Invalid session payload.' });
      return;
    }
    const current = (userSessionsStore.get(user.id) as Array<{ id: string }>) || [];
    const updated = [session, ...current.filter((s) => s.id !== session.id)].slice(0, 30);
    userSessionsStore.set(user.id, updated);
    savePersistedSessions();
    res.json({ ok: true });
  });

  app.delete('/api/sessions/:id', (req, res) => {
    const cookies = parseCookies(req.headers.cookie);
    const user = verifyUserCookie(cookies[COOKIE_NAME]);
    if (!user) {
      res.status(401).json({ error: 'Not signed in.' });
      return;
    }
    const targetId = req.params.id;
    const current = (userSessionsStore.get(user.id) as Array<{ id: string }>) || [];
    userSessionsStore.set(
      user.id,
      current.filter((s) => s.id !== targetId)
    );
    savePersistedSessions();
    res.json({ ok: true });
  });

  // Streaming Research Agent Pipeline
  app.post('/api/research/stream', async (req, res) => {
    const {
      query,
      depth = 'standard',
      timeRange = 'any',
      format = 'executive_brief',
      includeDomains = [],
      excludeDomains = [],
      attachments = [],
      model,
    } = req.body as {
      query?: string;
      depth?: 'quick' | 'standard' | 'deep';
      timeRange?: string;
      format?: string;
      includeDomains?: string[];
      excludeDomains?: string[];
      attachments?: AttachmentPayload[];
      model?: string;
    };

    if (!query || typeof query !== 'string' || !query.trim()) {
      res.status(400).json({ error: 'Please provide a research query.' });
      return;
    }

    const activeModel = resolveModelName(model);

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    let closed = false;
    res.on('close', () => {
      if (!res.writableEnded) {
        closed = true;
      }
    });

    const sendEvent = (event: string, data: unknown) => {
      if (closed || res.writableEnded) return;
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const heartbeat = setInterval(() => {
      if (!closed && !res.writableEnded) {
        res.write(': ping\n\n');
      }
    }, 8000);

    const startedAt = Date.now();

    try {
      const ai = getGeminiClient();
      const subQueryCount = depth === 'quick' ? 2 : depth === 'deep' ? 4 : 3;
      const cleanInclude = includeDomains
        .map((d) => d.trim().toLowerCase().replace(/^www\./, ''))
        .filter(Boolean);
      const cleanExclude = excludeDomains
        .map((d) => d.trim().toLowerCase().replace(/^www\./, ''))
        .filter(Boolean);
      const timeInstruction = getTimeRangeInstruction(timeRange);
      const formatInstruction = getFormatInstruction(format);
      const attachmentParts = buildAttachmentParts(attachments);

      // Step 1: Plan Research Queries
      const planStepId = 'step-plan';
      const planStart = Date.now();
      sendEvent('step', {
        id: planStepId,
        type: 'plan',
        title: 'Planning research vectors',
        detail: `Decomposing prompt into ${subQueryCount} targeted web search queries (${depth} depth).`,
        status: 'running',
        engine: 'gemini',
      });

      const planPrompt = [
        `You are an autonomous research planner. Today's date: ${new Date().toISOString().slice(0, 10)}.`,
        `Research query: "${query.trim()}"`,
        timeInstruction,
        cleanInclude.length > 0 ? `Prefer domains: ${cleanInclude.join(', ')}` : '',
        cleanExclude.length > 0 ? `Exclude domains: ${cleanExclude.join(', ')}` : '',
        `Create exactly ${subQueryCount} complementary, specific web search queries to thoroughly answer the user's question.`,
      ]
        .filter(Boolean)
        .join('\n');

      let plannedQueries: Array<{ id: string; query: string; angle: string }> = [
        { id: 'search-1', query: query.trim(), angle: 'Primary Search' },
      ];

      try {
        const planResp = await generateWithModelFallback(
          ai,
          'gemini-3.1-flash-lite',
          (modelName) => ({
            model: modelName,
            contents:
              attachmentParts.length > 0
                ? { parts: [...attachmentParts, { text: planPrompt }] }
                : planPrompt,
            config: {
              thinkingConfig: {
                thinkingLevel:
                  modelName === 'gemini-3.1-flash-lite'
                    ? ThinkingLevel.MINIMAL
                    : ThinkingLevel.LOW,
              },
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  queries: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        query: { type: Type.STRING },
                        angle: { type: Type.STRING },
                      },
                      required: ['query', 'angle'],
                    },
                  },
                },
                required: ['queries'],
              },
            },
          }),
          { maxAttempts: 2, timeoutMs: 9000 }
        );

        const parsed = JSON.parse((planResp.text ?? '{}').trim()) as {
          queries?: Array<{ query: string; angle: string }>;
        };
        if (parsed.queries && parsed.queries.length > 0) {
          plannedQueries = parsed.queries.slice(0, subQueryCount).map((q, idx) => ({
            id: `search-${idx + 1}`,
            query: q.query,
            angle: q.angle,
          }));
        }
      } catch (planErr) {
        const msg = planErr instanceof Error ? planErr.message : String(planErr);
        const isTransient =
          isQuotaOrUnavailableError(planErr) ||
          msg.includes('timed out') ||
          msg.includes('aborted') ||
          msg.includes('500');
        if (!isTransient) {
          throw planErr;
        }
        if (subQueryCount > 1) {
          plannedQueries.push({
            id: 'search-2',
            query: `${query.trim()} data statistics analysis`,
            angle: 'Supporting Evidence & Data',
          });
        }
      }

      sendEvent('step', {
        id: planStepId,
        type: 'plan',
        title: 'Research plan ready',
        detail: plannedQueries.map((q) => q.angle).join(' · '),
        status: 'completed',
        engine: 'gemini',
        durationMs: Date.now() - planStart,
      });

      // Step 2: Parallel Google Search Grounding
      const sourceMap = new Map<
        string,
        {
          id: number;
          title: string;
          url: string;
          domain: string;
          snippet?: string;
          extracted?: boolean;
          publishedDate?: string;
        }
      >();
      let nextSourceId = 1;

      const isAllowed = (domain: string) =>
        !cleanExclude.some((ex) => domain === ex || domain.endsWith(`.${ex}`));

      const registerSource = (
        url: string,
        title: string,
        snippet?: string,
        publishedDate?: string
      ) => {
        const normalized = url.trim();
        if (!normalized) return null;
        const domain = extractDomain(normalized, title);
        if (!isAllowed(domain)) return null;
        const existing = sourceMap.get(normalized);
        if (existing) return existing;
        const entry: {
          id: number;
          title: string;
          url: string;
          domain: string;
          snippet?: string;
          extracted?: boolean;
          publishedDate?: string;
        } = {
          id: nextSourceId++,
          title: title?.trim() || domain,
          url: normalized,
          domain,
          snippet,
          extracted: true,
          ...(publishedDate ? { publishedDate } : {}),
        };
        sourceMap.set(normalized, entry);
        return entry;
      };

      const searchFindings: Array<{ angle: string; query: string; text: string }> = [];

      await Promise.all(
        plannedQueries.map(async (sq) => {
          if (closed) return;
          const sqStart = Date.now();
          const usingTavily = Boolean(process.env.TAVILY_API_KEY);
          sendEvent('step', {
            id: sq.id,
            type: 'search',
            title: sq.angle,
            detail: sq.query,
            status: 'running',
            engine: usingTavily ? 'tavily' : 'gemini',
          });

          try {
            let countForQuery = 0;
            let bodyText = '';
            let usedEngine: 'tavily' | 'gemini' = usingTavily ? 'tavily' : 'gemini';

            // 1. Real Tavily Search API if configured
            if (usingTavily) {
              const tavilyRes = await fetchTavilyResults(
                sq.query,
                depth,
                timeRange,
                cleanInclude,
                cleanExclude
              );
              if (tavilyRes && tavilyRes.hits.length > 0) {
                for (const hit of tavilyRes.hits) {
                  const reg = registerSource(hit.url, hit.title, hit.snippet);
                  if (reg) {
                    if (hit.publishedDate) reg.publishedDate = hit.publishedDate;
                    countForQuery++;
                  }
                }
                bodyText = [
                  tavilyRes.answer ? `Summary: ${tavilyRes.answer}` : '',
                  ...tavilyRes.hits.map((h) => `- ${h.title} (${h.url}): ${h.snippet}`),
                ]
                  .filter(Boolean)
                  .join('\n');
              }
            }

            // 2. Try Gemini Google Search Grounding when Tavily not used or returned empty
            if (countForQuery === 0 && Date.now() >= googleSearchCooldownUntil) {
              usedEngine = 'gemini';
              const siteFilter =
                cleanInclude.length > 0
                  ? ` (${cleanInclude.slice(0, 4).map((d) => `site:${d}`).join(' OR ')})`
                  : '';

              try {
                const resp = await ai.models.generateContent({
                  model:
                    activeModel === 'gemini-4-hydrogen' && Date.now() < hydrogenCooldownUntil
                      ? 'gemini-3.1-flash-lite'
                      : activeModel,
                  contents: `Search and gather factual evidence for: "${sq.query}${siteFilter}"\nContext: "${query.trim()}"\n${timeInstruction}\n\nProvide dense, verified facts, metrics, dates, and entity details found in search results.`,
                  config: {
                    tools: [{ googleSearch: {} }],
                    thinkingConfig: {
                      thinkingLevel:
                        activeModel === 'gemini-3.1-flash-lite'
                          ? ThinkingLevel.MINIMAL
                          : depth === 'deep'
                          ? ThinkingLevel.MEDIUM
                          : ThinkingLevel.LOW,
                    },
                  },
                });

                const metadata = resp.candidates?.[0]?.groundingMetadata;
                const chunks =
                  (metadata?.groundingChunks as GroundingChunk[] | undefined) || [];
                const supports =
                  (metadata?.groundingSupports as GroundingSupport[] | undefined) || [];

                const snippetByChunkIdx = new Map<number, string>();
                for (const sup of supports) {
                  const segText = sup.segment?.text?.trim();
                  if (!segText || !sup.groundingChunkIndices) continue;
                  for (const idx of sup.groundingChunkIndices) {
                    if (!snippetByChunkIdx.has(idx)) {
                      snippetByChunkIdx.set(idx, segText);
                    }
                  }
                }

                bodyText = resp.text?.trim() || '';
                chunks.forEach((chunk, chunkIdx) => {
                  if (chunk.web?.uri) {
                    const snippet =
                      snippetByChunkIdx.get(chunkIdx) || bodyText.slice(0, 220) || undefined;
                    const reg = registerSource(
                      chunk.web.uri,
                      chunk.web.title || extractDomain(chunk.web.uri),
                      snippet
                    );
                    if (reg) countForQuery++;
                  }
                });
              } catch (groundingErr) {
                if (isQuotaOrUnavailableError(groundingErr)) {
                  googleSearchCooldownUntil = Date.now() + 180_000;
                } else {
                  throw groundingErr;
                }
              }
            }

            // 3. Live multi-source HTTP web retrieval if Search Grounding hit quota or returned 0 URLs
            if (countForQuery === 0) {
              const liveHits = await fetchLiveWebResults(sq.query, cleanInclude, cleanExclude);
              for (const hit of liveHits) {
                const reg = registerSource(hit.url, hit.title, hit.snippet);
                if (reg) {
                  if (hit.publishedDate) reg.publishedDate = hit.publishedDate;
                  countForQuery++;
                }
              }
              if (liveHits.length > 0) {
                bodyText = liveHits
                  .map((h) => `- ${h.title} (${h.url}): ${h.snippet}`)
                  .join('\n');
              }
            }

            if (bodyText) {
              searchFindings.push({
                angle: sq.angle,
                query: sq.query,
                text: bodyText,
              });
            }

            sendEvent('sources', Array.from(sourceMap.values()));
            sendEvent('step', {
              id: sq.id,
              type: 'search',
              title: sq.angle,
              detail: sq.query,
              status: 'completed',
              engine: usedEngine,
              resultsCount: countForQuery,
              durationMs: Date.now() - sqStart,
            });
          } catch (err: unknown) {
            sendEvent('step', {
              id: sq.id,
              type: 'search',
              title: sq.angle,
              detail: err instanceof Error ? err.message : 'Search failed',
              status: 'error',
              engine: 'gemini',
              durationMs: Date.now() - sqStart,
            });
          }
        })
      );

      if (closed) {
        if (!res.writableEnded) res.end();
        return;
      }

      // Step 3: Structured Report Synthesis
      const synthStepId = 'step-synth';
      const synthStart = Date.now();
      const allSources = Array.from(sourceMap.values());

      sendEvent('step', {
        id: synthStepId,
        type: 'synthesize',
        title: 'Synthesizing structured report',
        detail: `Cross-referencing ${allSources.length} verified sources into executive summary, findings, action plan, and evidence table.`,
        status: 'running',
        engine: 'gemini',
      });

      const sourceIndexText =
        allSources.length > 0
          ? allSources.map((s) => `[${s.id}] ${s.title} (${s.url})`).join('\n')
          : 'No external URLs returned.';

      const evidenceText = searchFindings
        .map((f) => `### ${f.angle} (${f.query})\n${f.text}`)
        .join('\n\n');

      const synthesisPrompt = [
        `Synthesize an authoritative, structured research report answering: "${query.trim()}"`,
        `Format guidance: ${formatInstruction}`,
        `Evidence gathered:\n${evidenceText}`,
        `Verified Sources Index:\n${sourceIndexText}`,
        `Rules:
- Cite claims inline using exact bracketed numbers like [1] or [2] matching the Verified Sources Index (1..${allSources.length}).
- Populate keyFindings with 3 to 5 clear, distinct takeaways and matching sourceIds.
- Populate recommendations with 3 to 5 actionable next steps (priority High/Medium/Low, ownerOrNextStep).
- Populate evidenceMatrix with 3 to 6 quantitative or comparative dimensions (dimension, summary, concise dataPoint).
- Write reportMarkdown with crisp headings (##), comparative Markdown tables where applicable, and inline citations [1], [2]. Do not repeat a raw reference list at the bottom of reportMarkdown.`,
      ].join('\n\n');

      const synthResp = await generateWithModelFallback(ai, activeModel, (modelName) => ({
        model: modelName,
        contents:
          attachmentParts.length > 0
            ? { parts: [...attachmentParts, { text: synthesisPrompt }] }
            : synthesisPrompt,
        config: {
          thinkingConfig: {
            thinkingLevel:
              modelName === 'gemini-3.1-flash-lite'
                ? ThinkingLevel.MINIMAL
                : depth === 'deep'
                ? ThinkingLevel.MEDIUM
                : ThinkingLevel.LOW,
          },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              executiveSummary: { type: Type.STRING },
              keyFindings: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    heading: { type: Type.STRING },
                    detail: { type: Type.STRING },
                    sourceIds: { type: Type.ARRAY, items: { type: Type.INTEGER } },
                  },
                  required: ['heading', 'detail', 'sourceIds'],
                },
              },
              recommendations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    action: { type: Type.STRING },
                    rationale: { type: Type.STRING },
                    priority: { type: Type.STRING },
                    ownerOrNextStep: { type: Type.STRING },
                  },
                  required: ['action', 'rationale', 'priority', 'ownerOrNextStep'],
                },
              },
              evidenceMatrix: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    dimension: { type: Type.STRING },
                    summary: { type: Type.STRING },
                    dataPoint: { type: Type.STRING },
                  },
                  required: ['dimension', 'summary', 'dataPoint'],
                },
              },
              reportMarkdown: { type: Type.STRING },
              openQuestions: { type: Type.ARRAY, items: { type: Type.STRING } },
              suggestedFollowUps: { type: Type.ARRAY, items: { type: Type.STRING } },
            },
            required: [
              'title',
              'executiveSummary',
              'keyFindings',
              'recommendations',
              'evidenceMatrix',
              'reportMarkdown',
              'openQuestions',
              'suggestedFollowUps',
            ],
          },
        },
      }));

      const rawReport = JSON.parse((synthResp.text ?? '{}').trim()) as {
        title?: string;
        executiveSummary?: string;
        keyFindings?: Array<{ heading: string; detail: string; sourceIds: number[] }>;
        recommendations?: Array<{
          action: string;
          rationale: string;
          priority: string;
          ownerOrNextStep: string;
        }>;
        evidenceMatrix?: Array<{ dimension: string; summary: string; dataPoint: string }>;
        reportMarkdown?: string;
        openQuestions?: string[];
        suggestedFollowUps?: string[];
      };

      const normalizedReport = {
        title: rawReport.title || query.trim(),
        executiveSummary: rawReport.executiveSummary || '',
        keyFindings: Array.isArray(rawReport.keyFindings) ? rawReport.keyFindings : [],
        recommendations: Array.isArray(rawReport.recommendations)
          ? rawReport.recommendations.map((rec, idx) => ({
              id: `rec-${Date.now()}-${idx}`,
              action: rec.action,
              rationale: rec.rationale,
              priority:
                rec.priority === 'High' || rec.priority === 'Low' ? rec.priority : 'Medium',
              ownerOrNextStep: rec.ownerOrNextStep,
              completed: false,
            }))
          : [],
        evidenceMatrix: Array.isArray(rawReport.evidenceMatrix) ? rawReport.evidenceMatrix : [],
        reportMarkdown: rawReport.reportMarkdown || '',
        openQuestions: Array.isArray(rawReport.openQuestions) ? rawReport.openQuestions : [],
        suggestedFollowUps: Array.isArray(rawReport.suggestedFollowUps)
          ? rawReport.suggestedFollowUps.slice(0, 3)
          : [],
      };

      sendEvent('step', {
        id: synthStepId,
        type: 'synthesize',
        title: 'Synthesis complete',
        detail: `Generated report with ${normalizedReport.keyFindings.length} key findings and ${normalizedReport.recommendations.length} action items.`,
        status: 'completed',
        engine: 'gemini',
        durationMs: Date.now() - synthStart,
      });

      sendEvent('complete', {
        report: normalizedReport,
        sources: allSources,
        engineUsed: activeModel,
        durationMs: Date.now() - startedAt,
      });
      if (!res.writableEnded) res.end();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Research execution failed.';
      sendEvent('error', { message });
      if (!res.writableEnded) {
        res.end();
      }
    } finally {
      clearInterval(heartbeat);
    }
  });

  // Follow-up endpoint
  app.post('/api/research/followup', async (req, res) => {
    try {
      const {
        question,
        report,
        sources = [],
        history = [],
        runLiveSearch = true,
        attachments = [],
        model,
      } = req.body as {
        question?: string;
        report?: {
          title?: string;
          executiveSummary?: string;
          reportMarkdown?: string;
          keyFindings?: Array<{ heading: string; detail: string }>;
          evidenceMatrix?: Array<{ dimension: string; summary: string; dataPoint: string }>;
        };
        sources?: Array<{
          id: number;
          title: string;
          url: string;
          domain: string;
          snippet?: string;
        }>;
        history?: Array<{ role: string; content: string }>;
        runLiveSearch?: boolean;
        attachments?: AttachmentPayload[];
        model?: string;
      };

      if (!question?.trim()) {
        res.status(400).json({ error: 'Follow-up question is required.' });
        return;
      }

      const activeModel = resolveModelName(model);
      const ai = getGeminiClient();
      const attachmentParts = buildAttachmentParts(attachments);

      const historyText = history
        .slice(-6)
        .map((h) => `${h.role === 'user' ? 'User' : 'Assistant'}: ${h.content}`)
        .join('\n');

      const existingSourcesText = sources
        .map((s) => `[${s.id}] ${s.title} (${s.url})`)
        .join('\n');

      const findingsSummary =
        report?.keyFindings && report.keyFindings.length > 0
          ? `Key Findings:\n${report.keyFindings.map((f, i) => `${i + 1}. ${f.heading}: ${f.detail}`).join('\n')}`
          : '';

      const matrixSummary =
        report?.evidenceMatrix && report.evidenceMatrix.length > 0
          ? `Metrics:\n${report.evidenceMatrix.map((m) => `- ${m.dimension}: ${m.dataPoint} (${m.summary})`).join('\n')}`
          : '';

      const promptText = [
        `Answer the user's follow-up question clearly and concisely with inline numeric citations [n].`,
        report
          ? `Existing Report Title: ${report.title}\nSummary: ${report.executiveSummary}\n${findingsSummary}\n${matrixSummary}\nDetails: ${(report.reportMarkdown || '').slice(0, 3500)}`
          : '',
        existingSourcesText ? `Existing Numbered Sources:\n${existingSourcesText}` : '',
        historyText ? `Recent Conversation:\n${historyText}` : '',
        `Follow-up question: "${question.trim()}"`,
        runLiveSearch
          ? 'Use live web search if needed for fresh facts; cite existing sources by their [n] number.'
          : 'Answer strictly from the existing report, sources, and any attached documents.',
      ]
        .filter(Boolean)
        .join('\n\n');

      const updatedSources = [...sources];
      const existingByUrl = new Map(updatedSources.map((s) => [s.url, s]));
      let nextId = updatedSources.reduce((m, s) => Math.max(m, s.id), 0) + 1;
      const citations: number[] = [];

      let resp;
      if (runLiveSearch) {
        if (Date.now() >= googleSearchCooldownUntil) {
          try {
            resp = await ai.models.generateContent({
              model:
                activeModel === 'gemini-4-hydrogen' && Date.now() < hydrogenCooldownUntil
                  ? 'gemini-3.1-flash-lite'
                  : activeModel,
              contents:
                attachmentParts.length > 0
                  ? { parts: [...attachmentParts, { text: promptText }] }
                  : promptText,
              config: {
                tools: [{ googleSearch: {} }],
                thinkingConfig: {
                  thinkingLevel:
                    activeModel === 'gemini-3.1-flash-lite'
                      ? ThinkingLevel.LOW
                      : ThinkingLevel.MEDIUM,
                },
              },
            });
          } catch (err) {
            if (!isQuotaOrUnavailableError(err)) {
              throw err;
            }
            googleSearchCooldownUntil = Date.now() + 180_000;
          }
        }

        if (!resp) {
          // Fall back to live HTTP web retrieval + model synthesis without grounding tool
          const liveHits = await fetchLiveWebResults(question.trim());
          const newSourceLines: string[] = [];
          for (const hit of liveHits) {
            let entry = existingByUrl.get(hit.url);
            if (!entry) {
              entry = {
                id: nextId++,
                title: hit.title,
                url: hit.url,
                domain: extractDomain(hit.url, hit.title),
                snippet: hit.snippet,
              };
              existingByUrl.set(hit.url, entry);
              updatedSources.push(entry);
            }
            if (!citations.includes(entry.id)) {
              citations.push(entry.id);
            }
            newSourceLines.push(`[${entry.id}] ${entry.title} (${entry.url}): ${hit.snippet}`);
          }

          const augmentedPrompt = [
            promptText,
            newSourceLines.length > 0
              ? `Fresh Live Web Evidence:\n${newSourceLines.join('\n')}`
              : '',
          ]
            .filter(Boolean)
            .join('\n\n');

          resp = await generateWithModelFallback(ai, activeModel, (modelName) => ({
            model: modelName,
            contents:
              attachmentParts.length > 0
                ? { parts: [...attachmentParts, { text: augmentedPrompt }] }
                : augmentedPrompt,
            config: {
              thinkingConfig: {
                thinkingLevel:
                  modelName === 'gemini-3.1-flash-lite'
                    ? ThinkingLevel.MINIMAL
                    : ThinkingLevel.LOW,
              },
            },
          }));
        }
      } else {
        resp = await generateWithModelFallback(ai, activeModel, (modelName) => ({
          model: modelName,
          contents:
            attachmentParts.length > 0
              ? { parts: [...attachmentParts, { text: promptText }] }
              : promptText,
          config: {
            thinkingConfig: {
              thinkingLevel:
                modelName === 'gemini-3.1-flash-lite'
                  ? ThinkingLevel.MINIMAL
                  : ThinkingLevel.LOW,
            },
          },
        }));
      }

      const answer = resp.text?.trim() || 'No response generated.';
      const chunks =
        (resp.candidates?.[0]?.groundingMetadata?.groundingChunks as
          | GroundingChunk[]
          | undefined) || [];

      for (const chunk of chunks) {
        if (chunk.web?.uri) {
          const uri = chunk.web.uri.trim();
          let entry = existingByUrl.get(uri);
          if (!entry) {
            entry = {
              id: nextId++,
              title: chunk.web.title || extractDomain(uri),
              url: uri,
              domain: extractDomain(uri, chunk.web.title),
            };
            existingByUrl.set(uri, entry);
            updatedSources.push(entry);
          }
          if (!citations.includes(entry.id)) {
            citations.push(entry.id);
          }
        }
      }

      // Also capture any existing [n] referenced in the answer
      const refMatches = answer.match(/\[(\d+)\]/g) || [];
      for (const m of refMatches) {
        const num = Number(m.slice(1, -1));
        if (updatedSources.some((s) => s.id === num) && !citations.includes(num)) {
          citations.push(num);
        }
      }

      res.json({
        answer,
        citations,
        sources: updatedSources,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Follow-up failed.';
      res.status(500).json({ error: msg });
    }
  });

  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      const indexFile = path.join(distPath, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Build output not found.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Tavily Research Agent server listening on http://localhost:${PORT}`);
  });
}

startServer();
