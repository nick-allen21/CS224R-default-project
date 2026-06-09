const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, ShadingType, PageNumber, Footer
} = require("/tmp/node_modules/docx");

const BLUE = "1F3864", BAR = "2E75B6", GREY = "595959", BOXBG = "F2F6FB", CODEBG = "F3F3F0";

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });

function body(runs, opts = {}) {
  const children = (typeof runs === "string") ? [new TextRun(runs)] : runs;
  return new Paragraph({ spacing: { after: 120 }, children, ...opts });
}
function bold(t) { return new TextRun({ text: t, bold: true }); }
function txt(t) { return new TextRun(t); }
function ital(t) { return new TextRun({ text: t, italics: true }); }
function code(t) { return new TextRun({ text: t, font: "Consolas", size: 19 }); }

function script(text) {
  return new Paragraph({
    spacing: { before: 60, after: 140 },
    indent: { left: 200 },
    shading: { type: ShadingType.CLEAR, fill: BOXBG },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: BAR, space: 10 } },
    children: [new TextRun({ text: text, italics: true, size: 23 })],
  });
}
function cue(text) {
  return new Paragraph({
    spacing: { after: 160 },
    indent: { left: 200 },
    children: [new TextRun({ text: "Delivery cue: ", bold: true, color: GREY, size: 19 }),
               new TextRun({ text: text, italics: true, color: GREY, size: 19 })],
  });
}
function codeLine(t) {
  return new Paragraph({
    spacing: { after: 0 },
    indent: { left: 200 },
    shading: { type: ShadingType.CLEAR, fill: CODEBG },
    children: [new TextRun({ text: t, font: "Consolas", size: 18 })],
  });
}
function bullet(runs, level = 0) {
  const children = (typeof runs === "string") ? [new TextRun(runs)] : runs;
  return new Paragraph({ numbering: { reference: "b", level }, spacing: { after: 60 }, children });
}

const children = [];

children.push(new Paragraph({ spacing: { after: 40 },
  children: [new TextRun({ text: "Poster Walkthrough — Speaker Notes", bold: true, size: 36, color: BLUE })] }));
children.push(new Paragraph({ spacing: { after: 40 },
  children: [new TextRun({ text: "Self-Generated Curriculum via Joint Conjecturer/Prover RLOO for Arithmetic Reasoning", italics: true, size: 22 })] }));
children.push(new Paragraph({ spacing: { after: 160 },
  children: [new TextRun({ text: "Finn Stäblein, Nicholas Allen  •  CS224R, Spring 2026", size: 20, color: GREY })] }));
children.push(new Paragraph({ spacing: { after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BAR, space: 2 } },
  children: [new TextRun({ text: "Target runtime ~2:30 at normal speech rate — leaves room for interruptions. Shaded bars are the words to say; grey notes are cues. Deeper answers to likely questions are in the Q&A at the end.", size: 20, color: GREY })] }));

children.push(H2("1. The problem  (~20s)"));
children.push(script("LLM reasoning training is bottlenecked by data. Once a model has consumed its fixed training set, the signal it extracts from each example decays — it has already mastered them. We wanted to test whether a small LLM with a deterministic verifier could train its own curriculum: generate new problems adapted to what the model currently can and can't do. We test this on Countdown — a small arithmetic puzzle where the verifier is just a Python function."));

children.push(H2("2. What we were trying to do  (~20s)"));
children.push(script("Two copies of Qwen 0.5B. One — the conjecturer — generates new problems. The other — the prover — tries to solve them. The verifier scores the prover's attempts. The conjecturer is rewarded only when the prover solves a problem at moderate difficulty — neither trivially easy nor impossible. Both models train jointly with RLOO. The idea: an adaptive curriculum that tracks the prover's frontier."));

children.push(H2("3. Getting the conjecturer to generate problems  (~30s — the interesting part)"));
children.push(script("The conjecturer starts from the same SFT checkpoint as the prover — but that model was trained to SOLVE problems: given a problem, output an equation. We needed the opposite: generate problems."));
children.push(script("Our first attempt was a chat-style prompt — three example problems and an instruction, 'generate a new problem, do not solve them.' It failed: the solve reflex is so strong the model just solved the few-shot examples. 25% parseable output."));
children.push(script("The fix was a completion-mode prompt that ends mid-tag — the prompt stops at '<problem> numbers=[' — so the model's first generated token has to be a digit. Format is enforced by structure, not by instruction-following. That took us to 95% parseable, 45% solvable. We used this in production for v6 and v9."));
children.push(script("We also tried training a dedicated SFT conjecturer on 720 problem-only examples: 100% parseable but 0% solvable. The small dataset taught surface format without internalizing the solvability constraint. The solve-trained model actually generates MORE solvable problems despite never being asked to generate — because it has seen thousands of complete solution traces."));
children.push(cue("If asked for the exact few-shot examples, see Q&A 1. The poster's ablation table (Section 8) has these four rows."));

children.push(H2("4. v6 vs v9  (~30s)"));
children.push(script("We ran two versions of the joint training."));
children.push(script("v6 is the minimal viable version: a binary reward — the conjecturer gets 1 if the prover solves between 5% and 85% of attempts, else 0 — pure conjecturer-generated problems, 25 training steps."));
children.push(script("v9 adds three regularizers. A shaped reward — a continuous tent function peaking at a 40% solve rate, so the conjecturer always has a gradient direction. 50% mixed data — half the prover's training problems come from the fixed Countdown dataset, anchoring it against over-specializing to the conjecturer's narrow distribution. And a length penalty that discourages rambling without hard truncation. 100 training steps. Both work, but they win on different metrics."));

children.push(H2("5. Results  (~30s)"));
children.push(script("Pass@1 — first-attempt accuracy: SFT is 0.244, v9 is 0.515 — more than double. Vanilla RLOO reaches 0.547, so we're comparable on accuracy — but v9 used about 13,000 update samples versus RLOO's ~82,000. That's the ~6×, measured in update samples."));
children.push(script("Pass@16 — best of 16 attempts, our diversity proxy: v6 hits 0.800, the best. v9 and SFT tie at 0.760. RLOO drops to 0.740 — below the untrained baseline. Vanilla RL fine-tuning loses sample diversity even after 80,000 update samples; joint training preserves it."));
children.push(script("So the headline is: joint conjecturer/prover RLOO is comparable to vanilla RLOO at ~6× fewer training samples, while preserving sample diversity."));
children.push(cue("The 6× is in update samples, not wall-clock (wall-clock is only ~1.4× on one GPU). If a TA presses on compute, see Q&A 2."));

children.push(H2("Optional close  (~10s)"));
children.push(script("The stable regime is narrow — conservative conjecturer hyperparameters and the three regularizers are all necessary. Future work: an adaptive difficulty band that shifts as the prover improves, and scaling to larger models."));

children.push(H1("Practice tip — five anchor sentences"));
children.push(body("Transitions are the hard part. Memorize the first sentence of each section; the rest follows."));
children.push(bullet([bold("1. "), txt("“LLM reasoning training is bottlenecked by data.”")]));
children.push(bullet([bold("2. "), txt("“Two copies of Qwen 0.5B.”")]));
children.push(bullet([bold("3. "), txt("“The conjecturer is initialized from the same SFT checkpoint as the prover.”")]));
children.push(bullet([bold("4. "), txt("“We ran two versions of the joint training.”")]));
children.push(bullet([bold("5. "), txt("“Pass@1 — first-attempt accuracy.”")]));

children.push(new Paragraph({ children: [], pageBreakBefore: true }));
children.push(H1("Anticipated Q&A"));

children.push(H2("Q1.  What are the few-shot examples in the conjecturer prompt?"));
children.push(body("Two prompts, two different few-shot sets."));
children.push(body([bold("Chat-style prompt"), txt(" (the failed first attempt, 25% parseable) — 3 examples, copied from the source Countdown dataset:")]));
children.push(codeLine("<problem> numbers=[85, 31, 37, 4], target=76 </problem>"));
children.push(codeLine("<problem> numbers=[79, 54, 4, 18], target=25 </problem>"));
children.push(codeLine("<problem> numbers=[3, 4, 6, 8], target=24 </problem>"));
children.push(body([txt("Wrapped with an instruction (“generate a new problem, do NOT solve them”) and ending with “"), code("Assistant: Let me create a new problem step by step."), txt("” — which leaves the model free to start in any direction.")]));
children.push(body([bold("Completion-style prompt"), txt(" (production, 95% parseable) — 6 examples, mixing 3- and 4-number variants and a range of targets, ending mid-token so the model must continue with a digit:")]));
["<problem> numbers=[85, 31, 37, 4], target=76 </problem>",
 "<problem> numbers=[79, 54, 4, 18], target=25 </problem>",
 "<problem> numbers=[3, 4, 6, 8], target=24 </problem>",
 "<problem> numbers=[42, 9, 11], target=53 </problem>",
 "<problem> numbers=[12, 7, 50, 2], target=64 </problem>",
 "<problem> numbers=[6, 25, 8], target=39 </problem>",
 "<problem> numbers=[          ← prompt ends here; model must continue"].forEach(l => children.push(codeLine(l)));
children.push(body([bold("Why these examples: "), txt("the first three are direct copies from the source dataset (asingh15/countdown_tasks_3to4) — known-solvable; the extra three add pattern variety (different number counts and target sizes); and the 3-vs-4-number mix keeps the model from biasing toward one length.")]));
children.push(body([bold("The structural trick: "), txt("the chat prompt lets the model choose its first token (and it chooses to solve); the completion prompt removes that freedom by ending at "), code("<problem> numbers=["), txt(" — the only viable continuation is a digit. Format is enforced by structure, not instruction-following.")]));

children.push(H2("Q2.  Why does v9 use 6× less compute?"));
children.push(body([bold("It's per update sample"), txt(" (sequences processed by the gradient update), not wall-clock time.")]));
children.push(body([bold("The math:")]));
children.push(bullet([bold("Vanilla RLOO (~80 steps): "), txt("128 problems/step × 8 rollouts = 1,024 samples/step × 80 ≈ 82,000 samples.")]));
children.push(bullet([bold("Joint v9 (100 steps): "), txt("16 problems/step (8 from C + 8 from the fixed set) × 8 rollouts = 128 samples/step × 100 ≈ 13,000 samples.")]));
children.push(bullet([bold("Ratio: "), txt("82,000 / 13,000 ≈ 6.3×.")]));
children.push(body([bold("Why v9's batches are smaller: "), txt("GPU memory. The joint trainer hot-swaps two policy models plus the conjecturer's update worker and importance-weight computation across vLLM/HF, so batches had to stay small to avoid the lm_head OOM that crashed earlier attempts. Vanilla RLOO manages one model, so it can use much bigger batches.")]));
children.push(body([bold("Honest caveat: "), txt("the 6× is in samples, not wall-clock. RLOO ran ~5 h; v9 ~3.5 h — only ~1.4× on the same H100. Joint training has 2× the vLLM worker swaps per step (4 vs 2), ~30–60 s each, which eats most of the sample-efficiency win.")]));
children.push(body([bold("If a TA asks:")]));
children.push(script("By update samples, v9 is ~6× more sample-efficient. By wall-clock on the same H100 it's closer to 1.5× — joint training's worker hot-swap overhead eats most of the win. Parallelizing C-sampling and P-sampling across two GPUs would also drop wall-clock ~3–4×."));
children.push(body([bold("Easy reframe to skip units entirely: "), ital("“v9 reaches comparable performance with substantially fewer training samples — the curriculum signal makes each sample more informative.”")]));

const doc = new Document({
  creator: "CS224R",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 220, after: 110 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 600, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1280, right: 1280, bottom: 1280, left: 1280 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Poster walkthrough notes — ", size: 16, color: GREY }),
                 new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(__dirname + "/poster_walkthrough_notes.docx", buf);
  console.log("wrote poster_walkthrough_notes.docx");
});
