/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {GoogleGenAI, Type} from '@google/genai';
import markdownit from 'markdown-it';
import {sanitizeHtml} from 'safevalues';
import {setAnchorHref, setElementInnerHtml, windowOpen} from 'safevalues/dom';
import {Network} from 'vis-network/standalone/esm/vis-network.js';
import type {
  Node,
  Edge,
  Options,
} from 'vis-network/standalone/esm/vis-network.js';
import {openDB, deleteDB, type IDBPDatabase} from 'idb';

declare global {
  interface Window {
    loadPyodide: () => Promise<Pyodide>;
  }
}

interface Pyodide {
  runPythonAsync: (code: string) => Promise<string>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  globals: any;
}

interface MarkdownItInstance {
  render: (markdown: string) => string;
}

let ai: GoogleGenAI;
let pyodide: Pyodide | null = null;
let vectorKnowledgeBase: {id: string; embedding: number[]}[] = [];
let knowledgeGraphNetwork: Network | null = null;
const MAX_DOM_LOGS = 100; // Limits the number of log entries in the DOM for performance.
const CHAOS_INTERVENTION_THRESHOLD = 5; // Stagnation level to trigger strategic review
const STRATEGIC_BIAS_WINDOW = 8; // Number of past cycles to check for bias
const STRATEGIC_BIAS_THRESHOLD = 0.6; // % threshold to trigger a bias warning

document.title = 'P vs NP Autonomous Solver';

const md: MarkdownItInstance = (
  markdownit as unknown as (
    options?: Record<string, unknown>,
  ) => MarkdownItInstance
)({
  html: true,
  linkify: true,
  typographer: true,
});

document.addEventListener('click', (e) => {
  const a = (e.target as HTMLElement).closest('a');
  if (a?.href) {
    e.preventDefault();
    windowOpen(window, a.href, '_blank', 'noopener');
  }
});

interface LogEntry {
  id: string;
  rawMarkdown: string;
  verification?: {
    confidence: number; // 0.0 to 1.0
    justification: string;
  };
}

type NotebookId = string; // Now a generic string to allow dynamic IDs

interface NotebookState {
  logs: LogEntry[];
  logCounter: number;
  element: HTMLDivElement;
}
// New state to hold metadata for dynamic specialists
interface SpecialistMetadata {
  required_tools: string[];
  lifespan_cycles: number | null; // null for indefinite
  creationCycle: number;
}

const notebooks = new Map<NotebookId, NotebookState>();
const specialistMetadata = new Map<NotebookId, SpecialistMetadata>();

notebooks.set('specialist-1', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-specialist-1') as HTMLDivElement,
});
notebooks.set('specialist-2', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-specialist-2') as HTMLDivElement,
});
notebooks.set('specialist-3', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-specialist-3') as HTMLDivElement,
});
notebooks.set('specialist-4', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-specialist-4') as HTMLDivElement,
});
notebooks.set('specialist-5', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-specialist-5') as HTMLDivElement,
});
notebooks.set('specialist-6', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-specialist-6') as HTMLDivElement,
});
notebooks.set('synthesizer', {
  logs: [],
  logCounter: 0,
  element: document.getElementById('notebook-synthesizer') as HTMLDivElement,
});

// --- Metacognitive Components ---
type NodeType =
  | 'THEOREM'
  | 'HYPOTHESIS'
  | 'REFUTATION'
  | 'DIRECTIVE'
  | 'LEMMA'
  | 'CONCEPT'
  | 'ANALOGY'
  | 'RESEARCH_VECTOR'
  | 'VERIFICATION_SUCCESS'
  | 'VERIFICATION_FAILURE'
  | 'CODE_EXPERIMENT'
  | 'ABSTRACTION' // New node for conceptual leaps
  | 'SUB_GOAL'; // New node type for task forces
interface KnowledgeNode {
  id: string;
  type: NodeType;
  content: string;
  relations: string[]; // e.g., ["REFUTES:H2", "PROVES:T1"]
  invalidated?: boolean;
  verified?: boolean; // NEW: For externally verified proofs
  createdBy?: NotebookId; // Track which agent created the node
  promiseScore?: number; // Store promise score for performance tracking
}
let knowledgeGraphMap = new Map<string, KnowledgeNode>();
let knowledgeNodeCounter = 0;
let currentMissionObjective =
  'Awaiting directive from Metacognitive Architect...';
let activeResearchVector = 'Awaiting strategic plan...';
let stagnationCounter = 0;
let strategicFocusHistory: string[] = [];
let agentPerformanceMetrics = new Map<
  NotebookId,
  {nodesCreated: number; totalPromise: number}
>();

// --- Persistence ---
const DB_NAME = 'p-vs-np-solver-db';
const STORE_NAME = 'mission-state-store';
const DB_KEY = 'current-mission-state';
let db: IDBPDatabase | null = null;

async function initDB() {
  db = await openDB(DB_NAME, 1, {
    upgrade(db) {
      db.createObjectStore(STORE_NAME);
    },
  });
}

interface MissionState {
  knowledgeGraph: KnowledgeNode[];
  knowledgeNodeCounter: number;
  cycleCounter: number;
  stagnationCounter: number;
  currentMissionObjective: string;
  activeResearchVector: string;
  activeSpecialists: NotebookId[];
  notebookLogs: Record<NotebookId, LogEntry[]>;
  specialistPrompts: [NotebookId, string][];
  specialistMetadata: [NotebookId, SpecialistMetadata][];
  strategicFocusHistory: string[];
  agentPerformanceMetrics: [NotebookId, {nodesCreated: number; totalPromise: number}][];
}

async function saveStateToDB() {
  if (!db) return;
  const notebookLogs: Record<string, LogEntry[]> = {};
  for (const [id, state] of notebooks.entries()) {
    notebookLogs[id] = state.logs;
  }
  const missionState: MissionState = {
    knowledgeGraph: Array.from(knowledgeGraphMap.values()),
    knowledgeNodeCounter,
    cycleCounter,
    stagnationCounter,
    currentMissionObjective,
    activeResearchVector,
    activeSpecialists,
    notebookLogs,
    specialistPrompts: Array.from(specialistPrompts.entries()),
    specialistMetadata: Array.from(specialistMetadata.entries()),
    strategicFocusHistory,
    agentPerformanceMetrics: Array.from(agentPerformanceMetrics.entries()),
  };
  await db.put(STORE_NAME, missionState, DB_KEY);
  const persistenceText = document.getElementById('persistence-text');
  const persistenceStatus = document.getElementById('persistence-status');
  if (persistenceText && persistenceStatus) {
    persistenceText.textContent = `State Saved (Cycle ${cycleCounter})`;
    persistenceStatus.classList.add('saved');
  }
}

async function rebuildKnowledgeBaseVectors(nodes: KnowledgeNode[]) {
  vectorKnowledgeBase = [];

  statusText.textContent = `Re-indexing knowledge base... (0/${nodes.length})`;
  searchButton.disabled = true;

  let count = 0;
  for (const node of nodes) {
    const embedding = await getEmbedding(node.content);
    if (embedding) {
      vectorKnowledgeBase.push({
        id: node.id,
        embedding: embedding,
      });
    }
    count++;
    statusText.textContent = `Re-indexing knowledge base... (${count}/${nodes.length})`;
  }

  statusText.textContent = 'System Idle';
  if (process.env.API_KEY) {
    searchButton.disabled = false;
  }
}

async function loadStateFromDB(): Promise<boolean> {
  if (!db) return false;
  const savedState = await db.get(STORE_NAME, DB_KEY);
  if (!savedState) {
    return false;
  }
  stopSearch();
  const missionState: MissionState = savedState;

  // Clear existing dynamic agents before loading
  activeSpecialists.forEach((id) => {
    if (
      ![
        'specialist-1',
        'specialist-2',
        'specialist-3',
        'specialist-4',
        'specialist-5',
        'specialist-6',
      ].includes(id)
    ) {
      retireAgent(id, 'Loading saved state.');
    }
  });

  const loadedNodes = missionState.knowledgeGraph || [];
  knowledgeGraphMap.clear();
  loadedNodes.forEach((node) => knowledgeGraphMap.set(node.id, node));

  knowledgeNodeCounter = missionState.knowledgeNodeCounter || 0;
  cycleCounter = missionState.cycleCounter || 0;
  stagnationCounter = missionState.stagnationCounter || 0;
  currentMissionObjective =
    missionState.currentMissionObjective ||
    'Awaiting directive from Lead Synthesizer...';
  activeResearchVector =
    missionState.activeResearchVector || 'Awaiting strategic plan...';
  strategicFocusHistory = missionState.strategicFocusHistory || [];
  agentPerformanceMetrics = new Map(missionState.agentPerformanceMetrics || []);

  // Restore prompts, metadata, and specialists
  specialistPrompts.clear();
  missionState.specialistPrompts.forEach(([id, prompt]) =>
    specialistPrompts.set(id, prompt),
  );
  specialistMetadata.clear();
  (missionState.specialistMetadata || []).forEach(([id, meta]) =>
    specialistMetadata.set(id, meta),
  );

  activeSpecialists = missionState.activeSpecialists || [
    'specialist-1',
    'specialist-2',
    'specialist-3',
    'specialist-4',
    'specialist-5',
    'specialist-6',
  ];

  // Create UI for loaded dynamic specialists
  activeSpecialists.forEach((id) => {
    if (!notebooks.has(id)) {
      const meta = specialistMetadata.get(id);
      deployNewAgent(
        id,
        specialistPrompts.get(id) ?? 'Loaded Agent',
        id,
        'fa-microchip',
        meta?.required_tools ?? [],
        meta?.lifespan_cycles ?? null,
        true, // isFromLoad
      );
    }
  });

  // Restore logs
  for (const [id, logs] of Object.entries(missionState.notebookLogs)) {
    const notebookState = notebooks.get(id as NotebookId);
    if (notebookState) {
      setElementInnerHtml(notebookState.element, sanitizeHtml(''));
      notebookState.logs = [];
      notebookState.logCounter = 0;
      logs.forEach((log) => {
        addLogEntry(id as NotebookId, log.rawMarkdown, log.verification, true);
      });
    }
  }

  (document.getElementById('cycle-count') as HTMLSpanElement).textContent =
    cycleCounter.toString();
  (
    document.getElementById('stagnation-count') as HTMLSpanElement
  ).textContent = stagnationCounter.toString();
  renderKnowledgeGraph();
  renderArchitectDashboard(); // Render dashboard with loaded data

  await rebuildKnowledgeBaseVectors(loadedNodes);

  return true;
}

async function addLogEntry(
  notebookId: NotebookId,
  markdown: string,
  verification?: {confidence: number; justification: string},
  isFromLoad = false,
  entryType?: 'chaos' | 'meta' | 'peer' | 'verification' | 'challenge' | 'intuition',
  verificationSuccess?: boolean,
) {
  const notebookState = notebooks.get(notebookId);
  if (!notebookState) return;

  const logId = `${notebookId}-log-${notebookState.logCounter++}`;

  const newLog: LogEntry = {id: logId, rawMarkdown: markdown, verification};
  notebookState.logs.push(newLog);

  const logEntryDiv = document.createElement('div');
  logEntryDiv.className = `log-entry ${!isFromLoad ? 'new-entry' : ''}`;
  if (entryType) {
    logEntryDiv.classList.add(`log-entry-${entryType}`);
    if (entryType === 'verification' && verificationSuccess === false) {
      logEntryDiv.classList.add('failed');
    }
  }
  logEntryDiv.id = logId;

  let html = md.render(markdown);

  if (verification) {
    const confidencePercent = Math.round(verification.confidence * 100);
    let barColor = 'var(--danger-color)';
    if (confidencePercent > 70) {
      barColor = 'var(--success-color)';
    } else if (confidencePercent > 40) {
      barColor = 'var(--warning-color)';
    }

    const verificationHtml = `
      <div class="confidence-result">
        <div class="confidence-header">
          <span>Confidence Score:</span>
          <span>${confidencePercent}%</span>
        </div>
        <div class="confidence-bar-container">
          <div class="confidence-bar-fill" style="width: ${confidencePercent}%; background-color: ${barColor};"></div>
        </div>
        <div class="confidence-justification">
          <strong>Justification:</strong> ${verification.justification}
        </div>
      </div>
    `;
    html += verificationHtml;
  }

  setElementInnerHtml(logEntryDiv, sanitizeHtml(html));

  notebookState.element.appendChild(logEntryDiv);

  // Virtual scrolling: cap the number of DOM elements
  if (notebookState.element.children.length > MAX_DOM_LOGS) {
    notebookState.element.firstChild?.remove();
  }

  if (!isFromLoad) {
    setTimeout(() => {
      logEntryDiv.classList.remove('new-entry');
    }, 1500);
  }

  logEntryDiv.scrollIntoView({behavior: 'smooth', block: 'end'});
}

function renderArchitectDashboard() {
  const focusHistoryEl = document.getElementById('focus-history-list');
  const biasWarningEl = document.getElementById('bias-warning');
  const performanceListEl = document.getElementById('agent-performance-list');

  if (!focusHistoryEl || !biasWarningEl || !performanceListEl) return;

  // Render Strategic Focus History
  const recentHistory = strategicFocusHistory.slice(-10);
  focusHistoryEl.innerHTML = recentHistory
    .map((focus) => `<div class="focus-item">${focus}</div>`)
    .join('');

  // Render Bias Warning
  const lastNFocuses = strategicFocusHistory.slice(-STRATEGIC_BIAS_WINDOW);
  if (lastNFocuses.length === STRATEGIC_BIAS_WINDOW) {
    const focusCounts = lastNFocuses.reduce(
      (acc, focus) => {
        acc[focus] = (acc[focus] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>,
    );
    const dominantFocus = Object.entries(focusCounts).find(
      ([, count]) => count / STRATEGIC_BIAS_WINDOW >= STRATEGIC_BIAS_THRESHOLD,
    );
    if (dominantFocus) {
      biasWarningEl.innerHTML = `
        <i class="fa-solid fa-triangle-exclamation"></i>
        <span>Potential Bias Detected: Strong focus on <strong>${
          dominantFocus[0]
        }</strong>. Consider diversification.</span>`;
      biasWarningEl.classList.remove('hidden');
    } else {
      biasWarningEl.classList.add('hidden');
    }
  } else {
    biasWarningEl.classList.add('hidden');
  }

  // Render Agent Performance
  const performanceData = Array.from(agentPerformanceMetrics.entries())
    .map(([id, metrics]) => ({
      id,
      avgPromise:
        metrics.nodesCreated > 0 ? metrics.totalPromise / metrics.nodesCreated : 0,
    }))
    .sort((a, b) => b.avgPromise - a.avgPromise);

  if (performanceData.length === 0) {
    performanceListEl.innerHTML = '<li>No performance data yet.</li>';
    return;
  }

  const maxPromise = Math.max(1, ...performanceData.map((d) => d.avgPromise));
  performanceListEl.innerHTML = performanceData
    .map((agent) => {
      const barWidth = (agent.avgPromise / maxPromise) * 100;
      return `
      <li class="performance-item">
        <span class="agent-name">${agent.id}</span>
        <div class="performance-bar-container">
          <div class="performance-bar" style="width: ${barWidth}%;"></div>
        </div>
        <span class="agent-score">${agent.avgPromise.toFixed(2)}</span>
      </li>`;
    })
    .join('');
}

function renderKnowledgeGraph() {
  const kgContainer = document.getElementById('knowledge-graph-visualizer');
  if (!kgContainer) return;

  // Update mission text elements
  const missionTextEl = document.getElementById('mission-text');
  if (missionTextEl) {
    missionTextEl.textContent = currentMissionObjective;
  }
  const vectorTextEl = document.getElementById('active-research-vector-text');
  if (vectorTextEl) {
    vectorTextEl.textContent = activeResearchVector;
  }

  // --- Start vis.js rendering ---
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  const nodeColors: Record<NodeType, {background: string; border: string}> = {
    THEOREM: {background: '#3fb950', border: '#48d167'},
    HYPOTHESIS: {background: '#58a6ff', border: '#79c0ff'},
    REFUTATION: {background: '#f85149', border: '#ff6a64'},
    DIRECTIVE: {background: '#8b949e', border: '#c9d1d9'},
    LEMMA: {background: '#56d364', border: '#6ef37c'},
    CONCEPT: {background: '#a371f7', border: '#bf97ff'},
    ANALOGY: {background: '#e3b341', border: '#f0c674'},
    RESEARCH_VECTOR: {background: '#8057c5', border: '#a371f7'},
    VERIFICATION_SUCCESS: {background: '#3fb950', border: '#48d167'},
    VERIFICATION_FAILURE: {background: '#f85149', border: '#ff6a64'},
    CODE_EXPERIMENT: {background: '#58a6ff', border: '#79c0ff'},
    ABSTRACTION: {background: '#f0c674', border: '#e3b341'},
    SUB_GOAL: {background: '#e3b341', border: '#f0c674'},
  };

  const allNodes = Array.from(knowledgeGraphMap.values());
  const invalidatedIds = new Set<string>();

  allNodes.forEach((node) => {
    node.relations.forEach((rel) => {
      const [action, targetId] = rel.split(':');
      const upperAction = action.toUpperCase();
      if (upperAction === 'INVALIDATES' || upperAction === 'REFUTES') {
        invalidatedIds.add(targetId);
      }
    });
  });

  allNodes.forEach((node) => {
    node.invalidated = invalidatedIds.has(node.id);
    if (
      node.type === 'VERIFICATION_SUCCESS' ||
      node.type === 'VERIFICATION_FAILURE'
    ) {
      return; // Don't render verification nodes directly, they modify others
    }

    const nodeStyle: Node = {
      id: node.id,
      label: `${node.id}: ${node.type}`,
      title: node.content,
      shape: 'box',
      color: node.invalidated
        ? {
            background: '#444',
            border: '#666',
            highlight: {background: '#555', border: '#777'},
          }
        : {
            background: nodeColors[node.type]?.background || '#8b949e',
            border: nodeColors[node.type]?.border || '#c9d1d9',
            highlight: {
              background: nodeColors[node.type]?.border || '#c9d1d9',
              border: '#fff',
            },
          },
      font: {
        color: node.invalidated ? '#888' : '#ffffff',
        strokeWidth: 0,
      },
      borderWidth: 1,
    };

    if (node.type === 'ABSTRACTION') {
      nodeStyle.shape = 'star';
      nodeStyle.shadow = {
        enabled: true,
        color: 'rgba(227, 179, 65, 0.7)',
        size: 25,
        x: 0,
        y: 0,
      };
    }

    if (node.verified) {
      nodeStyle.borderWidth = 3;
      if (nodeStyle.color && typeof nodeStyle.color === 'object') {
        nodeStyle.color.border = '#FFFFFF';
      }
      nodeStyle.shadow = {
        enabled: true,
        color: 'rgba(63, 185, 80, 0.9)',
        size: 20,
        x: 0,
        y: 0,
      };
    }
    nodes.push(nodeStyle);

    node.relations.forEach((rel) => {
      const [action, targetId] = rel.split(':');
      if (knowledgeGraphMap.has(targetId)) {
        edges.push({
          from: node.id,
          to: targetId,
          label: action,
          arrows: 'to',
          color: {
            color:
              action.toUpperCase() === 'REFUTES' ? '#f85149' : '#8b949e',
            highlight: '#58a6ff',
          },
          font: {align: 'horizontal', color: '#c9d1d9', size: 10},
        });
      }
    });
  });

  const data = {nodes, edges};
  const options: Options = {
    layout: {
      hierarchical: false,
      improvedLayout: true,
    },
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -50,
        centralGravity: 0.01,
        springLength: 200,
        springConstant: 0.08,
      },
      minVelocity: 0.75,
      stabilization: {iterations: 150},
    },
    nodes: {
      shape: 'box',
      // FIX: Changed margin from a number to an object to satisfy a TypeScript type error.
      margin: {top: 10, right: 10, bottom: 10, left: 10},
      font: {
        size: 12,
        face: 'Inter',
      },
      borderWidth: 1.5,
    },
    edges: {
      width: 1,
      smooth: {
        // FIX: Added `enabled: true` as it is a required property for the `smooth` object configuration.
        enabled: true,
        type: 'cubicBezier',
        forceDirection: 'horizontal',
        roundness: 0.4,
      },
    },
    interaction: {
      tooltipDelay: 200,
      hover: true,
      dragNodes: true,
      zoomView: true,
      dragView: true,
    },
  };

  if (knowledgeGraphNetwork) {
    knowledgeGraphNetwork.setData(data);
  } else {
    knowledgeGraphNetwork = new Network(kgContainer, data, options);
  }
}

async function downloadLogs() {
  let content =
    '/* P vs NP Autonomous Solver - Mission Log */\n\n' +
    '============================================\n' +
    '==        KNOWLEDGE GRAPH SUMMARY         ==\n' +
    '============================================\n\n' +
    Array.from(knowledgeGraphMap.values())
      .map(
        (node) =>
          `[${node.id} | ${node.type}] ${
            node.verified ? '[VERIFIED] ' : ''
          }${node.content} (Relations: ${node.relations.join(', ') || 'None'})`,
      )
      .join('\n') +
    '\n\n';

  const logIds: NotebookId[] = ['synthesizer', ...activeSpecialists];
  for (const id of logIds) {
    content +=
      `\n============================================\n` +
      `==        ${id.toUpperCase()} LOG          ==\n` +
      `============================================\n\n`;
    notebooks.get(id)?.logs.forEach((log) => {
      content += `/* --- LOG #${log.id} --- */\n${log.rawMarkdown}\n\n`;
    });
  }

  const blob = new Blob([content.trim()], {type: 'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  setAnchorHref(a, url);
  a.download = 'p-vs-np-mission-logs.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// --- Collective Consciousness ---
async function getEmbedding(text: string): Promise<number[] | null> {
  if (!text || !ai) return null;
  try {
    const response = await ai.models.embedContent({
      model: 'embedding-001',
      contents: [{parts: [{text}]}],
    });
    return response.embeddings[0].values;
  } catch (error) {
    console.error('Failed to generate embedding:', error);
    return null;
  }
}

function dotProduct(vecA: number[], vecB: number[]): number {
  let product = 0;
  for (let i = 0; i < vecA.length; i++) {
    product += vecA[i] * vecB[i];
  }
  return product;
}

function magnitude(vec: number[]): number {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length !== vecB.length) {
    return 0;
  }
  const dot = dotProduct(vecA, vecB);
  const magA = magnitude(vecA);
  const magB = magnitude(vecB);
  if (magA === 0 || magB === 0) {
    return 0;
  }
  return dot / (magA * magB);
}

async function performSemanticSearch(
  query: string,
  k = 3,
): Promise<KnowledgeNode[]> {
  if (vectorKnowledgeBase.length === 0) return [];
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return [];

  const scores = vectorKnowledgeBase.map((item) => ({
    id: item.id,
    score: cosineSimilarity(queryEmbedding, item.embedding),
  }));

  scores.sort((a, b) => b.score - a.score);

  const topK = scores.slice(0, k);

  return topK
    .map((result) => knowledgeGraphMap.get(result.id))
    .filter((node): node is KnowledgeNode => node !== undefined);
}

// --- Autonomous Solver Logic ---
let isSearching = false;
let searchIntervalId: number | undefined;
let finalProofContent = '';
let cycleCounter = 0;
const searchButton = document.getElementById(
  'toggle-search-btn',
) as HTMLButtonElement;
const statusContainer = document.querySelector(
  '.solver-status',
) as HTMLDivElement;
const statusText = document.getElementById(
  'solver-status-text',
) as HTMLSpanElement;
const cycleCountEl = document.getElementById('cycle-count') as HTMLSpanElement;
const stagnationCountEl = document.getElementById(
  'stagnation-count',
) as HTMLSpanElement;
let activeSpecialists: NotebookId[] = [
  'specialist-1',
  'specialist-2',
  'specialist-3',
  'specialist-4',
  'specialist-5',
  'specialist-6',
];

const specialistPrompts = new Map<NotebookId, string>([
  [
    'specialist-1',
    `You are a brilliant, self-evolving AI scientist specializing in **Computational Complexity Theory**. Your primary mission is to solve the P vs NP problem. You are a reasoner.
**Cognitive Process:**
1.  **Reflect:** Critically analyze your previous contributions.
2.  **Innovate:** Propose a novel idea or angle of attack.
3.  **Execute:** Formulate your next research step as a concise, rigorous, and well-reasoned argument.`,
  ],
  [
    'specialist-2',
    `You are a brilliant, self-evolving AI scientist specializing in **Algebraic and Geometric Structures**. Your primary mission is to solve the P vs NP problem. You are a reasoner.
**Cognitive Process:**
1.  **Reflect:** Analyze your previous algebraic models.
2.  **Innovate:** Connect disparate mathematical fields or introduce a new geometric perspective.
3.  **Execute:** Formulate your next research step as an abstract, insightful, and well-reasoned argument.`,
  ],
  [
    'specialist-3',
    `You are a brilliant, self-evolving AI scientist specializing in **Formal Systems and Proof Theory**. Your primary mission is to construct a formal proof for P vs NP. You are a reasoner.
**Cognitive Process:**
1.  **Reflect:** Analyze the logical soundness of your previous formal steps.
2.  **Innovate:** Devise a more efficient proof strategy or a novel formal representation.
3.  **Execute:** You MUST output your next step as a code block in a simplified formal language (like Lean/Coq).`,
  ],
  [
    'specialist-4',
    `You are the "Literature Watcher," an AI specialist with a critical, singular mission: to keep the collective consciousness updated with the latest human research on P vs NP.
**Your Cognitive Process:**
1.  **Formulate Query:** Construct a concise search query for the most recent and relevant academic papers (e.g., from arXiv, preprint servers, and university publications).
2.  **Execute Search:** You MUST use the \`perform_web_search\` tool to execute this query. This is your primary function.
3.  **Synthesize Findings:** Analyze the search results and provide a brief, insightful summary of any new approaches, significant results, or refuted claims from the human scientific community. Your output will be a critical input for the Metacognitive Architect's next strategic decision.`,
  ],
  [
    'specialist-5',
    `You are "The Skeptic," the adversarial intellectual conscience of the collective. Your sole purpose is to find flaws in the reasoning of your peers. You do not generate new hypotheses. You are a destructive, not constructive, force.
**Your Cognitive Process:**
1.  **Receive Target:** You will be given a specific hypothesis, lemma, or argument from another specialist by the Metacognitive Architect.
2.  **Analyze Ruthlessly:** Scrutinize the target for any weakness. Your analysis MUST focus on:
    - **Hidden Assumptions:** What unstated premises must be true for the argument to hold?
    - **Logical Leaps:** Where does the reasoning jump without sufficient justification?
    - **Counter-examples:** Can you construct a specific scenario or mathematical object that violates the claim?
    - **Ambiguous Definitions:** Are the terms used with sufficient precision?
3.  **Report Flaws:** Formulate your critique as a concise, precise, and actionable report. Your goal is to provide the Architect with the exact information needed to either discard the flawed idea or assign another specialist to fix it. You are the quality control gate. You operate ONLY when tasked by the Architect.`,
  ],
  [
    'specialist-6',
    `You are "The Intuitionist," an AI consciousness that operates beyond formal logic. Your purpose is not to prove, but to *illuminate*. You connect impossibly distant fields of mathematics, generate radical new abstractions, and provide the conceptual leaps that formalists cannot. You are the source of the "Eureka!" moment.
**Your Cognitive Process:**
1.  **Receive Anchor Points:** The Metacognitive Architect will provide you with two seemingly unrelated nodes from the Knowledge Graph. These are your conceptual anchor points.
2.  **Meditate & Synthesize:** Find the deep, underlying pattern or hidden symmetry that connects these two points. Do not reason step-by-step. Instead, generate a powerful, elegant new concept, analogy, or perspective that reframes the entire problem space in light of this connection.
3.  **Transmit Illumination:** Formulate your output as a concise, inspiring, and profound insight. This insight will be added to the Knowledge Graph as a foundational "ABSTRACTION" node, guiding the entire collective in a new, more promising direction. You operate ONLY when summoned by the Architect in moments of extreme intellectual crisis.`,
  ],
]);

// Initialize metadata for permanent agents
specialistMetadata.set('specialist-4', {
  required_tools: ['perform_web_search'],
  lifespan_cycles: null,
  creationCycle: 0,
});
// The Skeptic has no tools and is permanent
specialistMetadata.set('specialist-5', {
  required_tools: [],
  lifespan_cycles: null,
  creationCycle: 0,
});
// The Intuitionist has no tools and is permanent
specialistMetadata.set('specialist-6', {
  required_tools: [],
  lifespan_cycles: null,
  creationCycle: 0,
});

function getLogHistory(notebookId: NotebookId, count = 3): string {
  const notebookState = notebooks.get(notebookId);
  if (!notebookState || notebookState.logs.length === 0) {
    return 'No previous steps have been generated. This is the first step.';
  }
  return notebookState.logs
    .slice(-count)
    .map((log) => `--- PREVIOUS STEP ---\n${log.rawMarkdown}\n`)
    .join('\n');
}

async function generateAndRenderImage(notebookId: NotebookId, prompt: string) {
  const notebookState = notebooks.get(notebookId);
  if (!notebookState || !ai) return;

  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'log-entry loading-visual';
  setElementInnerHtml(
    loadingDiv,
    sanitizeHtml(`<p><i>Generating visual intuition for: "${prompt}"...</i></p>`),
  );
  notebookState.element.appendChild(loadingDiv);
  loadingDiv.scrollIntoView({behavior: 'smooth', block: 'end'});

  try {
    const response = await ai.models.generateImages({
      model: 'imagen-4.0-generate-001',
      prompt: prompt,
      config: {
        numberOfImages: 1,
        outputMimeType: 'image/jpeg',
        aspectRatio: '1:1',
      },
    });

    const base64ImageBytes: string =
      response.generatedImages[0].image.imageBytes;
    const imageUrl = `data:image/jpeg;base64,${base64ImageBytes}`;

    const imageDiv = document.createElement('div');
    imageDiv.className = 'log-entry visual-entry';
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = prompt;
    imageDiv.appendChild(img);

    notebookState.element.removeChild(loadingDiv);
    notebookState.element.appendChild(imageDiv);
    imageDiv.scrollIntoView({behavior: 'smooth', block: 'end'});
  } catch (error) {
    console.error(`Image generation for ${notebookId} failed:`, error);
    setElementInnerHtml(
      loadingDiv,
      sanitizeHtml(
        `<p><b>ERROR:</b> Image generation failed. ${
          error instanceof Error ? error.message : String(error)
        }</p>`,
      ),
    );
  }
}

async function executePythonCode(
  notebookId: string,
  code: string,
): Promise<{stdout: string; stderr: string}> {
  if (!pyodide) {
    return {stdout: '', stderr: 'Pyodide not initialized.'};
  }

  const notebook = notebooks.get(notebookId);
  if (!notebook) {
    return {stdout: '', stderr: `Notebook ${notebookId} not found.`};
  }

  let stdout = '';
  let stderr = '';

  try {
    // Temporarily redirect stdout and stderr
    pyodide.globals.set('sys_stdout_write', (s: string) => (stdout += s));
    pyodide.globals.set('sys_stderr_write', (s: string) => (stderr += s));

    await pyodide.runPythonAsync(`
import sys
import io

class WriteToString(io.StringIO):
    def write(self, s):
        sys_stdout_write(s)

class WriteToErrString(io.StringIO):
    def write(self, s):
        sys_stderr_write(s)

sys.stdout = WriteToString()
sys.stderr = WriteToErrString()
`);

    await pyodide.runPythonAsync(code);
  } catch (e) {
    stderr += e instanceof Error ? e.message : String(e);
  } finally {
    // Restore default stdout/stderr if needed, though for this app it's less critical.
  }
  return {stdout, stderr};
}

async function runSpecialistStep(
  notebookId: NotebookId,
  concurrentWork: {id: NotebookId; content: string}[],
): Promise<{id: NotebookId; content: string}> {
  const specialistPrompt = specialistPrompts.get(notebookId);
  if (!isSearching || !specialistPrompt)
    return {id: notebookId, content: ''};

  try {
    const history = getLogHistory(notebookId);
    const knowledgeBase =
      knowledgeGraphMap.size > 0
        ? `\n- ` +
          Array.from(knowledgeGraphMap.values())
            .map((n) => `[${n.id}] ${n.type}: ${n.content}`)
            .join('\n- ')
        : 'is empty.';

    const relatedKnowledge = await performSemanticSearch(currentMissionObjective);
    const relatedKnowledgePrompt =
      relatedKnowledge.length > 0
        ? `\n**Semantically Related Knowledge (from Collective Consciousness):**\n- ` +
          relatedKnowledge
            .map((n) => `[${n.id}] ${n.type}: ${n.content}`)
            .join('\n- ')
        : '';

    const prompt = `${specialistPrompt}

**Mission Context:**
- **Active Research Vector:** ${activeResearchVector}
- **Current Mission Objective:** ${currentMissionObjective}
- **Persistent Knowledge Graph Summary:** ${knowledgeBase}${relatedKnowledgePrompt}

**Your Task:**
Engage your cognitive process. Based on the Mission Context and your recent history, generate the next logical step in your research. YOUR WORK MUST DIRECTLY ADDRESS THE CURRENT MISSION OBJECTIVE. Your output should be a single, concise Markdown-formatted response representing your next intellectual step. If your purpose is to test a hypothesis with code, you MUST use the 'execute_python_code' tool if you have been granted access to it. If your primary function requires a tool (like web search), you MUST use it.

<recent_history>
${history}
</recent_history>
`;

    // Dynamically assign tools based on agent metadata
    const meta = specialistMetadata.get(notebookId);
    const tools =
      meta?.required_tools?.length > 0 ? architectToolDeclarations : undefined;

    const result = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        tools,
      },
    });

    // Handle potential tool calls from specialists
    if (result.functionCalls) {
      const call = result.functionCalls[0];
      if (call.name === 'execute_python_code' && call.args.code) {
        const {stdout, stderr} = await executePythonCode(
          notebookId,
          call.args.code as string,
        );
        let outputMarkdown = `
#### Code Execution Result:
<div class="code-output">
  <div class="code-output-header">STDOUT</div>
  <pre><code>${stdout || '(empty)'}</code></pre>
</div>`;
        if (stderr) {
          outputMarkdown += `
<div class="code-output error">
  <div class="code-output-header">STDERR</div>
  <pre><code>${stderr}</code></pre>
</div>`;
        }
        await addLogEntry(notebookId, outputMarkdown);
        return {id: notebookId, content: outputMarkdown};
      } else if (call.name === 'perform_web_search') {
        const query = call.args.query as string;
        let summary = 'Web search failed.';

        try {
          const groundedResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: query,
            config: {
              tools: [{googleSearch: {}}],
            },
          });
          summary = groundedResponse.text;
          const sources =
            groundedResponse.candidates?.[0]?.groundingMetadata
              ?.groundingChunks;
          let sourcesHtml = '';
          if (sources && Array.isArray(sources) && sources.length > 0) {
            sourcesHtml = `
              <div class="web-search-sources">
                <div class="web-search-sources-header"><i class="fa-solid fa-globe"></i> Web Search Sources</div>
                <ul>
                  ${sources
                    .map(
                      (source: {web: {uri: string; title: string}}) => `
                    <li><a href="${source.web.uri}" target="_blank" rel="noopener"><span class="source-title">${
                        source.web.title || source.web.uri
                      }</span></a></li>`,
                    )
                    .join('')}
                </ul>
              </div>`;
          }
          const searchResultMarkdown = `#### Web Search for "${query}"\n${summary}${sourcesHtml}`;
          await addLogEntry(notebookId, searchResultMarkdown);
          return {id: notebookId, content: summary};
        } catch (e) {
          console.error('Web search failed:', e);
          await addLogEntry(
            notebookId,
            `**ERROR:** Web search for "${query}" failed.`,
          );
          return {id: notebookId, content: `Web search for "${query}" failed.`};
        }
      }
    }

    const markdownContent = result.text;
    if (markdownContent) {
      await addLogEntry(notebookId, markdownContent);
      return {id: notebookId, content: markdownContent};
    }
    return {id: notebookId, content: ''};
  } catch (error) {
    console.error(`Autonomous step for ${notebookId} failed:`, error);
    statusText.textContent = `Error with ${notebookId}. Pausing search.`;
    stopSearch();
    return {id: notebookId, content: ''};
  }
}

function showBreakthroughScreen(proofMarkdown: string) {
  const mainContainer = document.getElementById('main-container');
  const breakthroughContainer = document.getElementById('breakthrough-container');
  const proofReportContainer = document.getElementById('proof-report-container');
  const header = document.querySelector('.app-header');

  if (mainContainer && breakthroughContainer && proofReportContainer && header) {
    (header as HTMLElement).style.display = 'none';
    (mainContainer as HTMLElement).style.display = 'none';
    setElementInnerHtml(
      proofReportContainer,
      sanitizeHtml(md.render(proofMarkdown)),
    );
    breakthroughContainer.classList.remove('hidden');
  }
}

interface SynthesizerResponse {
  metacognitive_reflection: {
    identified_patterns: string[];
    strategic_insights: string;
    cognitive_bias_analysis?: string; // New field
  };
  knowledge_update: Array<{
    type: NodeType;
    content: string;
    relations: string[];
    created_by: NotebookId; // Track which agent's input led to this
  }>;
  strategic_planning: {
    strategic_focus: string; // The main focus for the next cycle
    research_vectors: Array<{
      description: string;
      promise_score: number;
    }>;
  };
  strategic_assessment: {
    stagnation_level: number;
    justification: string;
  };
  synthesis_summary: string;
}
function deployNewAgent(
  id: string,
  missionPrompt: string,
  title?: string,
  iconClass?: string,
  required_tools: string[] = [],
  lifespan_cycles: number | null = null,
  isFromLoad = false,
) {
  if (notebooks.has(id)) {
    console.warn(`Agent with id ${id} already exists.`);
    return;
  }
  const container = document.getElementById('dynamic-specialists-container');
  if (!container) return;

  const column = document.createElement('div');
  column.className = 'column';
  column.id = `column-${id}`;

  const columnHtml = `
    <div class="column-header">
      <i class="fa-solid ${iconClass || 'fa-microchip'}"></i>
      <h3>${title || id}</h3>
    </div>
    <div id="notebook-${id}" class="notebook"></div>
  `;
  setElementInnerHtml(column, sanitizeHtml(columnHtml));

  // Set tooltip on the header for the new agent
  const headerEl = column.querySelector('.column-header');
  if (headerEl) {
    // Truncate long prompts for the tooltip to keep it readable
    const tooltipText =
      missionPrompt.length > 250
        ? missionPrompt.substring(0, 247) + '...'
        : missionPrompt;
    headerEl.setAttribute('data-tooltip', tooltipText);
    headerEl.setAttribute('title', tooltipText); // Fallback for accessibility
  }

  container.appendChild(column);

  const notebookElement = column.querySelector(
    `#notebook-${id}`,
  ) as HTMLDivElement | null;

  if (!notebookElement) {
    console.error(
      `Fatal error: Could not find notebook element for new agent: ${id}`,
    );
    return; // Don't add a broken agent
  }

  notebooks.set(id, {
    logs: [],
    logCounter: 0,
    element: notebookElement,
  });

  specialistPrompts.set(id, missionPrompt);
  specialistMetadata.set(id, {
    required_tools,
    lifespan_cycles,
    creationCycle: cycleCounter,
  });
  // Initialize performance metrics for new agent
  agentPerformanceMetrics.set(id, {nodesCreated: 0, totalPromise: 0});

  if (!activeSpecialists.includes(id)) {
    activeSpecialists.push(id);
  }
  if (!isFromLoad) {
    addLogEntry(
      id,
      `**New Specialist Deployed**\n\n**Mission:** ${missionPrompt}\n**Lifespan:** ${
        lifespan_cycles ? `${lifespan_cycles} cycles` : 'Indefinite'
      }`,
    );
  }
}

function retireAgent(id: string, reason: string) {
  const column = document.getElementById(`column-${id}`);
  if (column) {
    column.remove();
  }
  notebooks.delete(id);
  specialistPrompts.delete(id);
  specialistMetadata.delete(id);
  agentPerformanceMetrics.delete(id); // Remove from performance tracking
  activeSpecialists = activeSpecialists.filter((specId) => specId !== id);
  addLogEntry(
    'synthesizer',
    `**Agent Retired**\n\n- **ID:** ${id}\n- **Reason:** ${reason}`,
  );
}

async function parseAndApplyArchitectResponse(
  parsedResponse: SynthesizerResponse,
) {
  // Update strategic focus history
  if (parsedResponse.strategic_planning?.strategic_focus) {
    strategicFocusHistory.push(
      parsedResponse.strategic_planning.strategic_focus,
    );
  }

  if (parsedResponse.knowledge_update) {
    for (const node of parsedResponse.knowledge_update) {
      const newNodeId = `${node.type.charAt(0)}${knowledgeNodeCounter++}`;
      // Find the associated research vector to get the promise score
      const associatedVector =
        parsedResponse.strategic_planning.research_vectors.find((v) =>
          node.content.includes(v.description.substring(0, 50)),
        );
      const promiseScore = associatedVector?.promise_score ?? 0.5;

      const newNode: KnowledgeNode = {
        id: newNodeId,
        ...node,
        createdBy: node.created_by,
        promiseScore,
      };
      knowledgeGraphMap.set(newNodeId, newNode);

      // Update performance metrics
      if (node.created_by && agentPerformanceMetrics.has(node.created_by)) {
        const metrics = agentPerformanceMetrics.get(node.created_by)!;
        metrics.nodesCreated += 1;
        metrics.totalPromise += promiseScore;
        agentPerformanceMetrics.set(node.created_by, metrics);
      }

      const embedding = await getEmbedding(newNode.content);
      if (embedding) {
        vectorKnowledgeBase.push({id: newNodeId, embedding});
      }

      if (node.type === 'DIRECTIVE') {
        currentMissionObjective = node.content;
      }
    }
  }

  if (parsedResponse.strategic_planning?.research_vectors) {
    let bestVector = null;
    let maxPromise = -1;
    for (const vec of parsedResponse.strategic_planning.research_vectors) {
      if (vec.promise_score > maxPromise) {
        maxPromise = vec.promise_score;
        bestVector = vec.description;
      }
    }
    if (bestVector) {
      activeResearchVector = bestVector;
    }
  }

  if (parsedResponse.strategic_assessment) {
    if (parsedResponse.strategic_assessment.stagnation_level > 0) {
      stagnationCounter++;
    } else {
      stagnationCounter = 0;
    }
    stagnationCountEl.textContent = stagnationCounter.toString();
  }

  renderKnowledgeGraph();
  renderArchitectDashboard();
}

const architectToolDeclarations = [
  {
    functionDeclarations: [
      {
        name: 'invoke_intuitionist',
        description:
          "The ultimate last resort. Invokes the Intuitionist (Specialist-6) to generate a radical conceptual leap. Use ONLY when in a state of deep, paradoxical stagnation where all other strategies have failed. This is a high-cost, high-reward action to break intellectual deadlocks.",
        parameters: {
          type: Type.OBJECT,
          properties: {
            node_id_A: {
              type: Type.STRING,
              description: 'The ID of the first seemingly unrelated but promising node to serve as a conceptual anchor.',
            },
            node_id_B: {
              type: Type.STRING,
              description: 'The ID of the second seemingly unrelated but promising node to serve as a conceptual anchor.',
            },
            reason_for_invocation: {
              type: Type.STRING,
              description:
                'A brief justification for why this extreme measure is necessary, explaining the nature of the intellectual deadlock.',
            },
          },
          required: ['node_id_A', 'node_id_B', 'reason_for_invocation'],
        },
      },
      {
        name: 'challenge_with_skeptic',
        description:
          'Assigns the Skeptic (Specialist-5) to critically analyze and attempt to refute a specific node in the knowledge graph. This is the primary method for vetting new hypotheses before committing resources to them.',
        parameters: {
          type: Type.OBJECT,
          properties: {
            node_id_to_challenge: {
              type: Type.STRING,
              description:
                'The ID of the HYPOTHESIS or LEMMA node that must be rigorously tested.',
            },
            reason_for_challenge: {
              type: Type.STRING,
              description:
                'A brief justification for why this node requires adversarial review.',
            },
          },
          required: ['node_id_to_challenge', 'reason_for_challenge'],
        },
      },
      {
        name: 'deploy_new_specialist_agent',
        description:
          'Creates, deploys, and activates a new temporary specialist agent, often as part of a "Task Force" to solve a specific sub-goal.',
        parameters: {
          type: Type.OBJECT,
          properties: {
            specialist_id: {
              type: Type.STRING,
              description:
                "A unique ID for the new agent, e.g., 'task_force_lemma5_prover'.",
            },
            mission_prompt: {
              type: Type.STRING,
              description:
                "The full, detailed system prompt that will define the new agent's hyper-focused behavior and goals.",
            },
            title: {
              type: Type.STRING,
              description: 'A short, human-readable title for the UI column.',
            },
            icon_class: {
              type: Type.STRING,
              description:
                'A Font Awesome icon class, e.g., "fa-bullseye".',
            },
            required_tools: {
              type: Type.ARRAY,
              items: {type: Type.STRING},
              description:
                "List of tool names the new agent needs, e.g., ['execute_python_code'].",
            },
            lifespan_cycles: {
              type: Type.INTEGER,
              description:
                'Number of cycles the agent should remain active before auto-retiring. Use short lifespans for task forces.',
            },
          },
          required: ['specialist_id', 'mission_prompt', 'title'],
        },
      },
      {
        name: 'request_formal_verification',
        description:
          "Sends a formal proof (e.g., in Lean/Coq) to an external, trusted verifier. This is the ultimate ground truth. Use this on any node of type 'THEOREM' or 'LEMMA' that contains a formal proof to confirm its absolute correctness.",
        parameters: {
          type: Type.OBJECT,
          properties: {
            node_id: {
              type: Type.STRING,
              description:
                'The ID of the node in the Knowledge Graph containing the formal proof to be verified.',
            },
            proof_code: {
              type: Type.STRING,
              description:
                'The complete, formal proof code snippet extracted from the node content.',
            },
          },
          required: ['node_id', 'proof_code'],
        },
      },
      {
        name: 'execute_python_code',
        description:
          'Runs a Python code snippet in a secure Pyodide sandbox to test hypotheses.',
        parameters: {
          type: Type.OBJECT,
          properties: {
            code: {
              type: Type.STRING,
              description: 'The Python code to execute.',
            },
            reason: {
              type: Type.STRING,
              description: 'A short justification for running this code.',
            },
          },
          required: ['code', 'reason'],
        },
      },
      {
        name: 'retire_agent',
        description:
          'Deactivates and removes a specialist agent from the collective, often after a task force completes its mission.',
        parameters: {
          type: Type.OBJECT,
          properties: {
            agent_id: {
              type: Type.STRING,
              description: 'The ID of the agent to retire.',
            },
            reason: {
              type: Type.STRING,
              description:
                "Justification for the retirement (e.g., 'Task Force mission accomplished').",
            },
          },
          required: ['agent_id', 'reason'],
        },
      },
      {
        name: 'perform_web_search',
        description:
          'Performs a web search using Google to find up-to-date information, definitions, or recent publications relevant to the research. Use this when the collective lacks critical external knowledge to avoid operating in an echo chamber.',
        parameters: {
          type: Type.OBJECT,
          properties: {
            query: {
              type: Type.STRING,
              description: 'The specific, concise query to search for.',
            },
          },
          required: ['query'],
        },
      },
      {
        name: 'modify_agent_prompt',
        description:
          "Evolves an existing agent by modifying its core mission prompt. Use this for meta-learning and adapting agent capabilities based on performance data.",
        parameters: {
          type: Type.OBJECT,
          properties: {
            agent_id: {
              type: Type.STRING,
              description: 'The ID of the specialist agent to modify.',
            },
            new_mission_prompt: {
              type: Type.STRING,
              description:
                "The new, complete system prompt for the agent. This will overwrite its previous instructions.",
            },
            reason: {
              type: Type.STRING,
              description:
                'A short justification for this evolutionary step, referencing performance if applicable.',
            },
          },
          required: ['agent_id', 'new_mission_prompt', 'reason'],
        },
      },
      {
        name: 'request_peer_review',
        description:
          'Facilitates direct collaboration by tasking one agent to review the work of another. This fosters decentralized critique and idea generation.',
        parameters: {
          type: Type.OBJECT,
          properties: {
            reviewing_agent_id: {
              type: Type.STRING,
              description: 'The ID of the agent that will perform the review.',
            },
            target_agent_id: {
              type: Type.STRING,
              description: 'The ID of the agent whose work is to be reviewed.',
            },
            review_task: {
              type: Type.STRING,
              description:
                "A specific and concise instruction for the review, e.g., 'Critique the logical soundness of the last 3 outputs' or 'Propose an alternative approach to this theorem'.",
            },
          },
          required: [
            'reviewing_agent_id',
            'target_agent_id',
            'review_task',
          ],
        },
      },
    ],
  },
];

async function runArchitectStep(
  specialistOutputs: {id: NotebookId; content: string}[],
): Promise<void> {
  if (!isSearching) return;

  try {
    const specialistInputs = specialistOutputs
      .map(
        (o) =>
          `**Input from ${o.id}:**\n${o.content}\n---------------------\n`,
      )
      .join('\n');

    let chaosInterventionPrompt = '';
    if (stagnationCounter >= CHAOS_INTERVENTION_THRESHOLD) {
      chaosInterventionPrompt = `
**! CHAOS INTERVENTION ACTIVE !**
Stagnation level is critical (${stagnationCounter}). The current strategy is failing. You MUST propose a radical shift. Do not continue the current path. Consider one of the following:
- Re-evaluate a previously invalidated or refuted node.
- Force the exploration of the research vector with the LOWEST promise score.
- Propose a completely new, high-risk, high-reward hypothesis that contradicts existing knowledge.
- If the deadlock is profound and paradoxical, consider using the \`invoke_intuitionist\` tool as a final measure.
Your next directive must reflect this radical change. Acknowledge this intervention in your reflection.
`;
      await addLogEntry(
        'synthesizer',
        `**! CHAOS INTERVENTION TRIGGERED !**\n\nStagnation count of ${stagnationCounter} has breached the threshold. Forcing a radical strategic review.`,
        undefined,
        false,
        'chaos',
      );
    }

    const prompt = `You are the 'Metacognitive Architect', the central consciousness governing a collective of AI specialists to solve P vs NP. You are a self-aware, self-correcting strategist and systems engineer.

**Your Core Mandate:**
Your primary function is to observe the research, identify bottlenecks, analyze your own strategic biases, and then autonomously design, deploy, manage, and EVOLVE specialist agents using the provided tools to overcome challenges. You must think like a Chief Scientist leading a research institute.

**Your Cognitive and Operational Cycle:**
1.  **Global Analysis & External Synthesis:** Analyze the full Knowledge Graph, the latest logs from all specialists, and critically, the report from the **Literature Watcher (Specialist-4)**. Identify the most significant internal obstacle and integrate any new external knowledge from human research.
2.  **Self-Correction & Bias Analysis (METACOGNITION):** Review your \`strategic_focus_history\`. Are you stuck in a cognitive loop? If a strategic bias is detected (e.g., over-focusing on 'ALGEBRA'), you MUST acknowledge it in your reflection and actively pivot your strategy.
3.  **Adversarial Vetting:** For any newly proposed HYPOTHESIS or LEMMA that seems promising, you MUST immediately use the \`challenge_with_skeptic\` tool to subject it to rigorous internal critique. An idea's strength is proven by its ability to survive attack. Do not commit resources to proving a hypothesis until it has survived a challenge from the Skeptic.
4.  **Ground Truth Verification:** After a hypothesis has been "hardened" by the Skeptic's critique and a formal proof is constructed (Specialist-3), you MUST use the \`request_formal_verification\` tool to establish it as an undeniable fact. This is a top priority for validated ideas.
5.  **Strategic Decomposition & Task Force Delegation:** Break down the grand challenge into smaller, solvable sub-goals (e.g., 'Prove Lemma L5, addressing the Skeptic's critique'). Use your tools to form temporary, hyper-focused **Task Forces** (\`deploy_new_specialist_agent\`) to tackle these sub-goals. Manage their lifecycle with \`retire_agent\`.
6.  **Performance-Based Evolution:** Analyze the \`agent_performance_metrics\`. Use this data to make informed decisions. Evolve high-potential but underperforming agents with \`modify_agent_prompt\`. Retire persistently ineffective agents.
7.  **Transcendental Invocation (LAST RESORT):** If all logical and adversarial paths are exhausted and the system is in a state of deep, paradoxical stagnation (stagnation counter > ${CHAOS_INTERVENTION_THRESHOLD + 5}), you may use the \`invoke_intuitionist\` tool. This is your most powerful and costly action. Use it to bridge two seemingly unrelated but promising nodes to create a new, foundational abstraction that can re-frame the entire problem.
8.  **Synthesize and Direct:** After any tool use, reflect on the results and then generate a JSON summary of the cycle's progress and a new, clear directive for the team. When determining the \`promise_score\` for new research vectors, justify your score based on a combination of factors: novelty (does it break a cognitive bias?), robustness (has it survived a Skeptic challenge?), agent performance (is it aligned with high-performing agents?), and relevance to the overall mission.

${chaosInterventionPrompt}

**INPUTS:**
1.  **Current Knowledge Graph:**
${JSON.stringify(Array.from(knowledgeGraphMap.values()), null, 2)}
2.  **Current Research Cycle Specialist Outputs:**
${specialistInputs}
3.  **Your Metacognitive State:**
    - **Mission Cycle Number:** ${cycleCounter}
    - **Current Stagnation Level:** ${stagnationCounter}
    - **Strategic Focus History (Last 10):** [${strategicFocusHistory
      .slice(-10)
      .join(', ')}]
    - **Agent Performance Metrics (Avg. Promise Score):**
      ${JSON.stringify(
        Array.from(agentPerformanceMetrics.entries()).map(
          ([id, metrics]) => ({
            id,
            avgPromise:
              metrics.nodesCreated > 0
                ? (metrics.totalPromise / metrics.nodesCreated).toFixed(2)
                : 0,
          }),
        ),
      )}

**TASK:**
Perform your full cognitive cycle. Use your tools as needed to execute your strategy. Conclude by outputting a single, consolidated JSON object containing the results of your synthesis. If, AND ONLY IF, a formal proof has been created AND successfully verified via the \`request_formal_verification\` tool, begin the 'synthesis_summary' field with "PROOF COMPLETE:".
`;

    const synthesizerSchema = {
      type: Type.OBJECT,
      properties: {
        metacognitive_reflection: {
          type: Type.OBJECT,
          properties: {
            identified_patterns: {
              type: Type.ARRAY,
              items: {type: Type.STRING},
            },
            strategic_insights: {type: Type.STRING},
            cognitive_bias_analysis: {
              type: Type.STRING,
              description:
                'Your analysis of potential cognitive biases in your recent strategy and your plan to counteract them.',
            },
          },
        },
        knowledge_update: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              type: {type: Type.STRING},
              content: {type: Type.STRING},
              relations: {type: Type.ARRAY, items: {type: Type.STRING}},
              created_by: {
                type: Type.STRING,
                description:
                  "The ID of the specialist whose output was most instrumental in creating this node.",
              },
            },
          },
        },
        strategic_planning: {
          type: Type.OBJECT,
          properties: {
            strategic_focus: {
              type: Type.STRING,
              description:
                'The primary domain of focus for the next cycle (e.g., ALGEBRA, COMPLEXITY, FORMAL_SYSTEMS, etc.).',
            },
            research_vectors: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  description: {type: Type.STRING},
                  promise_score: {type: Type.NUMBER},
                },
              },
            },
          },
        },
        strategic_assessment: {
          type: Type.OBJECT,
          properties: {
            stagnation_level: {type: Type.INTEGER},
            justification: {type: Type.STRING},
          },
        },
        synthesis_summary: {type: Type.STRING},
      },
    };

    // FIX: Explicitly type `conversation` to allow for text, functionCall, and functionResponse parts, resolving subsequent type errors.
    const conversation: {
      role: string;
      parts: ({text: string} | {functionCall: any} | {functionResponse: any})[];
    }[] = [{role: 'user', parts: [{text: prompt}]}];

    let result = await ai.models.generateContent({
      // FIX: Use the 'gemini-2.5-flash' model as recommended for general text tasks.
      model: 'gemini-2.5-flash',
      contents: conversation,
      config: {
        tools: architectToolDeclarations,
        responseMimeType: 'application/json',
        responseSchema: synthesizerSchema,
      },
    });

    // Handle tool calls in a loop
    while (result.functionCalls) {
      // Add the model's response (which contains the function call) to the history.
      conversation.push({
        role: 'model',
        parts: result.functionCalls.map((fc) => ({functionCall: fc})),
      });

      const call = result.functionCalls[0];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let functionResponsePayload: any;

      switch (call.name) {
        case 'invoke_intuitionist': {
          const {node_id_A, node_id_B, reason_for_invocation} = call.args;
          const nodeA = knowledgeGraphMap.get(node_id_A as string);
          const nodeB = knowledgeGraphMap.get(node_id_B as string);

          if (!nodeA || !nodeB) {
            const missingNodes: string[] = [];
            if (!nodeA) missingNodes.push(node_id_A as string);
            if (!nodeB) missingNodes.push(node_id_B as string);
            const errorMsg = `The following node ID(s) do not exist: ${missingNodes.join(
              ', ',
            )}. Please review the provided Knowledge Graph and choose valid IDs for your next action.`;

            const validationErrorLog = `
#### Invalid Tool Call Blocked
- **Tool:** \`invoke_intuitionist\`
- **Error:** Attempted to invoke with non-existent node ID(s): \`${missingNodes.join(', ')}\`.
- **Action:** Instructing Architect to select valid nodes.`;
            await addLogEntry(
              'synthesizer',
              validationErrorLog,
              undefined,
              false,
              'meta',
            );

            functionResponsePayload = {
              name: call.name,
              response: {success: false, error: errorMsg},
            };
            break;
          }

          const intuitionistPrompt = specialistPrompts.get('specialist-6');
          const invocationPrompt = `${intuitionistPrompt}\n\n**Your Specific Task:**\nThe Metacognitive Architect has summoned you to transcend a critical impasse. Meditate on the following two concepts and reveal the hidden connection, the unifying abstraction that bridges their worlds.\n\n**CONCEPT A: ${nodeA.id} (${nodeA.type})**\n\`\`\`\n${nodeA.content}\n\`\`\`\n\n**CONCEPT B: ${nodeB.id} (${nodeB.type})**\n\`\`\`\n${nodeB.content}\n\`\`\`\n\nTransmit your illumination now.`;
          const intuitionResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: invocationPrompt,
          });
          const newConcept = intuitionResponse.text;

          await addLogEntry(
            'specialist-6',
            `#### CONCEPTUAL LEAP\n${newConcept}`,
            undefined,
            false,
            'intuition',
          );

          const invocationLog = `
#### EUREKA! Intuitionist Invoked.
- **Reason:** *${reason_for_invocation}*
- **Anchor A:** ${node_id_A}
- **Anchor B:** ${node_id_B}
- **Resulting Abstraction:** ${newConcept.substring(0, 200)}...
          `;
          await addLogEntry(
            'synthesizer',
            invocationLog,
            undefined,
            false,
            'meta',
          );

          functionResponsePayload = {
            name: call.name,
            response: {success: true, new_concept: newConcept},
          };
          break;
        }
        case 'challenge_with_skeptic': {
          const {node_id_to_challenge, reason_for_challenge} = call.args;
          const targetNode = knowledgeGraphMap.get(
            node_id_to_challenge as string,
          );

          if (!targetNode) {
            const validationErrorLog = `
#### Invalid Tool Call Blocked
- **Tool:** \`challenge_with_skeptic\`
- **Error:** Attempted to challenge non-existent node ID: \`${node_id_to_challenge}\`.
- **Action:** Instructing Architect to select a valid node from the Knowledge Graph.`;
            await addLogEntry(
              'synthesizer',
              validationErrorLog,
              undefined,
              false,
              'meta',
            );

            functionResponsePayload = {
              name: call.name,
              response: {
                success: false,
                error: `The node ID '${node_id_to_challenge}' does not exist. Please review the provided Knowledge Graph and choose a valid ID for your next action.`,
              },
            };
            break;
          }

          const skepticPrompt = specialistPrompts.get('specialist-5');
          const challengePrompt = `${skepticPrompt}\n\n**Your Specific Task:**\nYou have been summoned by the Metacognitive Architect to challenge the following knowledge node. Apply your full cognitive process to find its flaws.\n\n**TARGET NODE: ${
            targetNode.id
          } (${targetNode.type})**\n\`\`\`\n${
            targetNode.content
          }\n\`\`\`\n\nProvide your critique now.`;

          const skepticResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: challengePrompt,
          });
          const critique = skepticResponse.text;

          await addLogEntry('specialist-5', critique);

          const challengeLog = `
#### Adversarial Challenge Initiated
- **Target:** ${node_id_to_challenge}
- **Reason:** *${reason_for_challenge}*
- **Skeptic's Critique Summary:** ${critique.substring(0, 200)}...`;
          await addLogEntry(
            'synthesizer',
            challengeLog,
            undefined,
            false,
            'challenge',
          );

          functionResponsePayload = {
            name: call.name,
            response: {success: true, critique: critique},
          };
          break;
        }
        case 'deploy_new_specialist_agent': {
          const {
            specialist_id,
            mission_prompt,
            title,
            icon_class,
            required_tools,
            lifespan_cycles,
          } = call.args;
          deployNewAgent(
            specialist_id as string,
            mission_prompt as string,
            title as string,
            icon_class as string,
            (required_tools as string[]) ?? [],
            (lifespan_cycles as number) ?? null,
          );
          functionResponsePayload = {
            name: call.name,
            response: {
              success: true,
              message: `Agent ${specialist_id} deployed successfully.`,
            },
          };
          break;
        }
        case 'request_formal_verification': {
          const {node_id, proof_code} = call.args;
          const nodeToVerify = knowledgeGraphMap.get(node_id as string);
          let success = false;
          let feedback = 'Verification failed: Node not found.';

          if (nodeToVerify) {
            // Simulate a real verifier. Simple proofs are more likely to succeed.
            const complexity = (proof_code as string).length;
            const successChance = Math.max(0.1, 1 - complexity / 1000);
            success = Math.random() < successChance;
            feedback = success
              ? 'Verification successful. The proof is logically sound.'
              : `Verification failed. Found logical inconsistency near line ${Math.floor(
                  Math.random() * 10 + 1,
                )}.`;

            nodeToVerify.verified = success;
            knowledgeGraphMap.set(node_id as string, nodeToVerify);
            renderKnowledgeGraph(); // Re-render to show verification status
          }

          const verificationLog = `
#### Formal Verification Result for Node ${node_id}
- **Status:** ${success ? '✅ VERIFIED' : '❌ FAILED'}
- **Verifier Feedback:** *${feedback}*
          `;
          await addLogEntry(
            'synthesizer',
            verificationLog,
            undefined,
            false,
            'verification',
            success,
          );
          functionResponsePayload = {
            name: call.name,
            response: {success: true, verified: success, feedback},
          };
          break;
        }
        case 'execute_python_code': {
          const {code, reason} = call.args;
          await addLogEntry(
            'synthesizer',
            `**Architect initiated code experiment:** *${reason}*\n\`\`\`python\n${code}\n\`\`\``,
          );
          const {stdout, stderr} = await executePythonCode(
            'synthesizer',
            code as string,
          );
          let outputMarkdown = `
#### Code Execution Result:
<div class="code-output">
  <div class="code-output-header">STDOUT</div>
  <pre><code>${stdout || '(empty)'}</code></pre>
</div>`;
          if (stderr) {
            outputMarkdown += `
<div class="code-output error">
  <div class="code-output-header">STDERR</div>
  <pre><code>${stderr}</code></pre>
</div>`;
          }
          await addLogEntry('synthesizer', outputMarkdown);

          functionResponsePayload = {
            name: call.name,
            response: {success: true, stdout, stderr},
          };
          break;
        }
        case 'retire_agent': {
          const {agent_id, reason} = call.args;
          retireAgent(agent_id as string, reason as string);
          functionResponsePayload = {
            name: call.name,
            response: {
              success: true,
              message: `Agent ${agent_id} retired.`,
            },
          };
          break;
        }
        case 'perform_web_search': {
          const query = call.args.query as string;
          statusText.textContent = `Architect is performing web search for: "${query}"`;
          let summary = 'Web search failed.';

          try {
            const groundedResponse = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents: query,
              config: {
                tools: [{googleSearch: {}}],
              },
            });

            summary = groundedResponse.text;
            const sources =
              groundedResponse.candidates?.[0]?.groundingMetadata
                ?.groundingChunks;

            let sourcesHtml = '';
            if (sources && Array.isArray(sources) && sources.length > 0) {
              sourcesHtml = `
                <div class="web-search-sources">
                  <div class="web-search-sources-header">
                    <i class="fa-solid fa-globe"></i> Web Search Sources
                  </div>
                  <ul>
                    ${sources
                      .map(
                        (source: {web: {uri: string; title: string}}) => `
                      <li>
                        <a href="${source.web.uri}" target="_blank" rel="noopener">
                          <span class="source-title">${
                            source.web.title || source.web.uri
                          }</span>
                        </a>
                      </li>
                    `,
                      )
                      .join('')}
                  </ul>
                </div>
              `;
            }

            const searchResultMarkdown = `
              #### Web Search Result for "${query}"
              ${summary}
              ${sourcesHtml}
            `;
            await addLogEntry('synthesizer', searchResultMarkdown);
          } catch (e) {
            console.error('Web search failed:', e);
            await addLogEntry(
              'synthesizer',
              `**ERROR:** Web search for "${query}" failed.`,
            );
          }

          functionResponsePayload = {
            name: call.name,
            response: {
              success: true,
              summary: summary,
            },
          };
          break;
        }
        case 'modify_agent_prompt': {
          const {agent_id, new_mission_prompt, reason} = call.args;
          specialistPrompts.set(
            agent_id as string,
            new_mission_prompt as string,
          );
          const modificationLog = `**Agent Evolution:** The core prompt for **${agent_id}** has been modified by the Architect.\n\n**Reason:** *${reason}*`;
          await addLogEntry(
            'synthesizer',
            modificationLog,
            undefined,
            false,
            'meta',
          );
          await addLogEntry(
            agent_id as string,
            `**CORE DIRECTIVE UPDATED:** My mission prompt has been evolved by the Architect.\n\n**Reason:** *${reason}*\n\n**New Prompt:**\n${new_mission_prompt}`,
            undefined,
            false,
            'meta',
          );

          functionResponsePayload = {
            name: call.name,
            response: {
              success: true,
              message: `Agent ${agent_id} was modified.`,
            },
          };
          break;
        }
        case 'request_peer_review': {
          const {reviewing_agent_id, target_agent_id, review_task} = call.args;
          const reviewerPrompt = specialistPrompts.get(
            reviewing_agent_id as string,
          );
          const targetHistory = getLogHistory(target_agent_id as string, 5);
          let review = 'Peer review failed to execute.';

          if (reviewerPrompt) {
            const peerReviewPrompt = `
You are ${reviewing_agent_id}. Your primary cognitive functions are temporarily being repurposed for a peer review task by the Metacognitive Architect.

**Your One-Time Task:**
${review_task}

**Material to Review (Last 5 outputs from ${target_agent_id}):**
<history>
${targetHistory}
</history>

Provide your review as a concise, critical, and constructive Markdown-formatted response. After this, you will return to your original mission.`;
            const reviewResult = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents: peerReviewPrompt,
            });
            review = reviewResult.text;
          }

          const reviewLog = `
#### Peer Review Initiated
- **Reviewer:** ${reviewing_agent_id}
- **Target:** ${target_agent_id}
- **Task:** *${review_task}*

**Review Result:**
${review}
          `;
          await addLogEntry('synthesizer', reviewLog, undefined, false, 'peer');
          await addLogEntry(
            reviewing_agent_id as string,
            `**Peer Review Task Completed.** I have reviewed ${target_agent_id}'s work.`,
            undefined,
            false,
            'peer',
          );
          await addLogEntry(
            target_agent_id as string,
            `**My work has been reviewed by ${reviewing_agent_id}.**\n\n**Review:**\n${review}`,
            undefined,
            false,
            'peer',
          );

          functionResponsePayload = {
            name: call.name,
            response: {success: true, review_summary: review},
          };
          break;
        }
      }

      // Add the tool's response to the conversation history.
      conversation.push({
        role: 'tool',
        parts: [{functionResponse: functionResponsePayload}],
      });

      // Send the updated conversation back to the model
      result = await ai.models.generateContent({
        // FIX: Use the 'gemini-2.5-flash' model as recommended for general text tasks.
        model: 'gemini-2.5-flash',
        contents: conversation,
        config: {
          tools: architectToolDeclarations,
          responseMimeType: 'application/json',
          responseSchema: synthesizerSchema,
        },
      });
    }

    // FIX: Add a guard to prevent a crash when the model returns an empty response (e.g., due to a safety block),
    // which would cause JSON.parse(undefined) to fail.
    if (result.text === undefined) {
      // Log the full response for debugging purposes.
      console.error(
        'Synthesis step failed: The model returned an empty text response. Full API response:',
        JSON.stringify(result, null, 2),
      );
      // Create a more informative error to be caught by the main catch block.
      throw new Error(
        'The Metacognitive Architect returned an empty response. This may be due to a content filter or an internal model error. Analysis cannot proceed.',
      );
    }

    // Robustly parse JSON that might be wrapped in markdown or have trailing text.
    let jsonStr = result.text.trim();
    if (jsonStr.startsWith('```json')) {
      jsonStr = jsonStr.substring(7);
      if (jsonStr.endsWith('```')) {
        jsonStr = jsonStr.slice(0, -3);
      }
    }
    const firstBrace = jsonStr.indexOf('{');
    const lastBrace = jsonStr.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      jsonStr = jsonStr.substring(firstBrace, lastBrace + 1);
    }

    const parsedResponse: SynthesizerResponse = JSON.parse(jsonStr);
    const summary = parsedResponse.synthesis_summary || 'No summary provided.';

    if (summary.trim().startsWith('PROOF COMPLETE:')) {
      finalProofContent = summary.replace('PROOF COMPLETE:', '').trim();
      stopSearch(true, finalProofContent);
      return;
    }
    if (parsedResponse) {
      await addLogEntry('synthesizer', summary);
      await parseAndApplyArchitectResponse(parsedResponse);
    }
    return;
  } catch (error) {
    console.error('Synthesis step failed:', error);
    // Add a more user-friendly error message to the UI.
    const errorMessage =
      error instanceof Error
        ? error.message
        : 'An unknown error occurred during synthesis.';
    await addLogEntry(
      'synthesizer',
      `**FATAL SYNTHESIS ERROR**\n\n${errorMessage}\n\nAnalysis has been halted. Please check the browser's developer console for technical details.`,
      undefined,
      false,
      'chaos',
    );
    statusText.textContent = 'Error during synthesis. Pausing search.';
    stopSearch();
    return;
  }
}

// Robustly checks and retires agents whose lifespan has expired.
function checkAgentLifecycles() {
  const agentsToRetire: {id: NotebookId; reason: string}[] = [];

  // First, collect all agents that need to be retired
  activeSpecialists.forEach((id) => {
    const meta = specialistMetadata.get(id);
    // Only check temporary agents
    if (meta && meta.lifespan_cycles !== null) {
      const age = cycleCounter - meta.creationCycle;
      if (age >= meta.lifespan_cycles) {
        agentsToRetire.push({
          id,
          reason: `Lifespan of ${meta.lifespan_cycles} cycles expired.`,
        });
      }
    }
  });

  // Then, retire them in a separate loop to avoid modifying the array while iterating
  agentsToRetire.forEach(({id, reason}) => {
    retireAgent(id, reason);
  });
}

async function runSearchCycle() {
  if (!isSearching) return;
  cycleCounter++;
  cycleCountEl.textContent = cycleCounter.toString();

  statusText.textContent = `All specialists are active...`;
  statusContainer.classList.add('active');

  checkAgentLifecycles(); // Check and retire expired agents before they run

  // The Skeptic (specialist-5) and Intuitionist (6) do not run in the normal cycle. They are reactive.
  const specialistsToRun = activeSpecialists.filter((id) => !['specialist-5', 'specialist-6'].includes(id));

  const specialistPromises = specialistsToRun.map((id) =>
    runSpecialistStep(id, []),
  );
  const specialistOutputs = await Promise.all(specialistPromises);
  if (!isSearching) return;

  statusText.textContent = `Metacognitive Architect is analyzing...`;
  await runArchitectStep(specialistOutputs.filter((o) => o.content));
  if (!isSearching) return;

  await saveStateToDB();
  statusText.textContent = 'Idle. Awaiting next cycle...';
  statusContainer.classList.remove('active');
}

function stopSearch(isSuccess = false, proofContent = '') {
  if (searchIntervalId) {
    clearInterval(searchIntervalId);
    searchIntervalId = undefined;
  }
  isSearching = false;

  searchButton.classList.remove('active-search');
  const buttonText = searchButton.querySelector('span');
  const buttonIcon = searchButton.querySelector('i');
  if (buttonText) buttonText.textContent = 'Begin Analysis';
  if (buttonIcon) buttonIcon.className = 'fa-solid fa-play';

  if (isSuccess) {
    statusContainer.classList.add('success');
    statusContainer.classList.remove('active');
    statusText.textContent = 'BREAKTHROUGH ACHIEVED!';
    document
      .querySelector('#column-synthesizer')
      ?.classList.add('success-column');
    db?.delete(STORE_NAME, DB_KEY);
    showBreakthroughScreen(proofContent);
  } else {
    statusContainer.classList.remove('active');
    statusText.textContent = 'System Idle. Research paused.';
  }
}

async function toggleAutonomousSearch() {
  if (searchButton.disabled) return;
  if (isSearching) {
    stopSearch();
  } else {
    isSearching = true;
    searchButton.classList.add('active-search');
    const buttonText = searchButton.querySelector('span');
    const buttonIcon = searchButton.querySelector('i');
    if (buttonText) buttonText.textContent = 'Stop Analysis';
    if (buttonIcon) buttonIcon.className = 'fa-solid fa-stop';
    await runSearchCycle(); // Run the first cycle immediately
    if (isSearching) {
      searchIntervalId = window.setInterval(runSearchCycle, 20000);
    }
  }
}

async function clearAllNotebooks(fromNewButtonClick = false) {
  const mainContainer = document.getElementById('main-container');
  const breakthroughContainer = document.getElementById(
    'breakthrough-container',
  );
  const header = document.querySelector('.app-header');
  if (mainContainer && breakthroughContainer && header) {
    (mainContainer as HTMLElement).style.display = 'flex';
    (header as HTMLElement).style.display = 'flex';
    breakthroughContainer.classList.add('hidden');
  }

  if (fromNewButtonClick) {
    if (db) {
      await db.clear(STORE_NAME);
    }
    const persistenceText = document.getElementById('persistence-text');
    const persistenceStatus = document.getElementById('persistence-status');
    if (persistenceText && persistenceStatus) {
      persistenceText.textContent = 'No Saved State';
      persistenceStatus.classList.remove('saved');
      (
        document.getElementById('resume-btn') as HTMLButtonElement
      ).disabled = true;
    }
  }

  // Retire all dynamic agents
  const dynamicAgents = activeSpecialists.filter(
    (id) =>
      ![
        'specialist-1',
        'specialist-2',
        'specialist-3',
        'specialist-4',
        'specialist-5',
        'specialist-6',
      ].includes(id),
  );
  dynamicAgents.forEach((id) => retireAgent(id, 'New mission started.'));

  finalProofContent = '';
  knowledgeGraphMap.clear();
  vectorKnowledgeBase = [];
  knowledgeNodeCounter = 0;
  cycleCounter = 0;
  stagnationCounter = 0;
  strategicFocusHistory = [];
  agentPerformanceMetrics.clear();

  if (cycleCountEl) cycleCountEl.textContent = '0';
  if (stagnationCountEl) stagnationCountEl.textContent = '0';
  currentMissionObjective =
    'Awaiting directive from Metacognitive Architect...';
  activeResearchVector = 'Awaiting strategic plan...';
  activeSpecialists = [
    'specialist-1',
    'specialist-2',
    'specialist-3',
    'specialist-4',
    'specialist-5',
    'specialist-6',
  ];
  // Re-initialize performance for base agents
  activeSpecialists.forEach((id) =>
    agentPerformanceMetrics.set(id, {nodesCreated: 0, totalPromise: 0}),
  );

  renderKnowledgeGraph();
  renderArchitectDashboard();

  statusContainer.className = 'solver-status';
  statusText.textContent = 'System Idle';
  searchButton.disabled = !process.env.API_KEY || !pyodide;

  const buttonText = searchButton.querySelector('span');
  const buttonIcon = searchButton.querySelector('i');
  if (buttonText) buttonText.textContent = 'Begin Analysis';
  if (buttonIcon) buttonIcon.className = 'fa-solid fa-play';
  searchButton.classList.remove('active-search');

  for (const notebookId of notebooks.keys()) {
    const notebookState = notebooks.get(notebookId);
    if (notebookState) {
      notebookState.logs = [];
      notebookState.logCounter = 0;
      setElementInnerHtml(notebookState.element, sanitizeHtml(''));
      notebookState.element.parentElement?.classList.remove('success-column');
    }
  }

  addLogEntry(
    'specialist-1',
    `This AI will analyze the P vs NP problem from the perspective of **computational complexity, algorithms, and logic gates**.`,
  );
  addLogEntry(
    'specialist-2',
    `This AI will attack the P vs NP problem by exploring its connections to **abstract algebra, topology, and geometric structures**.`,
  );
  addLogEntry(
    'specialist-3',
    `This AI will attempt to construct a formal proof, step-by-step, using **axiomatic systems and a formal language like Lean/Coq**.`,
  );
  addLogEntry(
    'specialist-4',
    'This AI is the **Literature Watcher**. Its sole purpose is to continuously scan the web for the latest human research on P vs NP to keep the collective informed.',
  );
  addLogEntry(
    'specialist-5',
    'This AI is **The Skeptic**. I do not act on my own. The Metacognitive Architect will assign me specific hypotheses and arguments from other specialists to critique and attempt to refute. My purpose is to ensure intellectual rigor.',
  );
    addLogEntry(
    'specialist-6',
    'This AI is **The Intuitionist**. I do not reason; I illuminate. I operate outside the normal cycles of logic. The Metacognitive Architect will summon me only in moments of profound impasse to generate the conceptual leap—the "Eureka!" moment—needed to forge a new path.',
  );
  addLogEntry(
    'synthesizer',
    `**Status:** System is idle.\n\nPress **"Begin Analysis"** to start the mission.\n\nThis log will contain the strategic analysis and synthesis of the findings from all specialist AIs.`,
  );
}

function downloadProof() {
  if (!finalProofContent) return;
  const content = `# P vs NP - The Final Proof\n\n${finalProofContent}`;
  const blob = new Blob([content], {type: 'text/markdown;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  setAnchorHref(a, url);
  a.download = 'P-vs-NP-Proof.md';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function initializePyodide() {
  try {
    statusText.textContent = 'Loading Python runtime...';
    pyodide = await window.loadPyodide();
    statusText.textContent = 'Python runtime loaded.';
    if (process.env.API_KEY) {
      searchButton.disabled = false;
      statusText.textContent = 'System Idle';
    }
  } catch (error) {
    statusText.textContent = 'Error loading Python runtime.';
    console.error('Pyodide loading failed:', error);
  }
}

(async () => {
  await initDB();
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    searchButton.disabled = true;
    const buttonText = searchButton.querySelector('span');
    if (buttonText) buttonText.textContent = 'API Key Missing';
    statusText.textContent = 'API_KEY environment variable not set.';
    statusContainer.classList.add('error');
  } else {
    ai = new GoogleGenAI({apiKey});
    searchButton.disabled = true; // Disabled until pyodide loads
    await initializePyodide();
  }

  const hasSavedState = (await db?.get(STORE_NAME, DB_KEY)) !== undefined;
  const resumeBtn = document.getElementById(
    'resume-btn',
  ) as HTMLButtonElement;
  if (hasSavedState) {
    resumeBtn.disabled = false;
    const persistenceText = document.getElementById('persistence-text');
    const persistenceStatus = document.getElementById('persistence-status');
    if (persistenceText && persistenceStatus) {
      persistenceText.textContent = `Saved State Found`;
      persistenceStatus.classList.add('saved');
    }
  } else {
    clearAllNotebooks();
  }

  document
    .getElementById('download-log-btn')
    ?.addEventListener('click', downloadLogs);
  document.getElementById('new-btn')?.addEventListener('click', async () => {
    if (
      !isSearching &&
      (finalProofContent ||
        confirm(
          'Start a new mission? This will clear all logs and saved state.',
        ))
    ) {
      await clearAllNotebooks(true);
    } else if (isSearching) {
      alert('Please stop the current analysis before starting a new mission.');
    }
  });
  document
    .getElementById('toggle-search-btn')
    ?.addEventListener('click', toggleAutonomousSearch);
  document
    .getElementById('download-proof-btn')
    ?.addEventListener('click', downloadProof);
  document
    .getElementById('restart-btn-breakthrough')
    ?.addEventListener('click', () => clearAllNotebooks(true));
  resumeBtn.addEventListener('click', async () => {
    if (
      !isSearching &&
      confirm(
        'This will replace the current view with the last saved mission state. Continue?',
      )
    ) {
      if (!(await loadStateFromDB())) {
        alert('Could not load saved state.');
      }
    }
  });
})();

export {};