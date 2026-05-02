# MCP Memory Service - Demo Recording Script

**Duration Target:** 30 seconds max
**Output:** High-quality screen recording for GIF conversion
**Goal:** Show dramatic before/after difference

---

## 🎬 Recording Setup

### Before You Start

1. **Close unnecessary apps** (clean desktop)
2. **Set screen resolution** to 1920x1080 or 1280x720
3. **Disable notifications** (Do Not Disturb mode)
4. **Prepare two Claude Desktop sessions:**
   - Session A: Fresh session WITHOUT memory context
   - Session B: Fresh session WITH memory enabled
5. **Pre-populate memory** with project context:
   ```bash
   # Use /memory-store to add context before recording
   claude /memory-store "Project: Next.js app with App Router, Prisma ORM, tRPC for API, TypeScript, Tailwind CSS"
   claude /memory-store "Architecture decision: Using server components for data fetching, client components only when needed"
   claude /memory-store "Current focus: Building authentication system with NextAuth.js"
   ```

### Recording Tools

**macOS (Recommended):**
```bash
# Option 1: QuickTime Player
# File → New Screen Recording → Click red record button → Select area

# Option 2: Built-in Screenshot.app
# Cmd+Shift+5 → Record Selected Portion → Select Claude window → Record

# Option 3: ffmpeg (for automation)
screencapture -v demo-raw.mov
```

**Windows:**
- Windows + G (Game Bar)
- OBS Studio
- ScreenToGif

---

## 🎭 Scene 1: WITHOUT Memory (10 seconds)

**Setup:** Fresh Claude Desktop session, memory service disabled

### Actions:

1. **[0:00-0:02]** Show empty Claude chat
   - Visible: "New conversation" at top

2. **[0:02-0:05]** Type and send:
   ```
   What tech stack am I using for my current project?
   ```

3. **[0:05-0:10]** Claude responds (expected):
   ```
   I don't have any information about your current project's
   tech stack. Could you tell me what technologies you're using?
   ```

4. **[0:10]** **Visual indicator:** Red ❌ or "Context Lost" overlay

---

## 🎭 Scene 2: WITH Memory (20 seconds)

**Setup:** Fresh Claude Desktop session, memory service enabled & populated

### Actions:

1. **[0:10-0:12]** Show NEW empty Claude chat (different session)
   - Visible: "New conversation" at top
   - **Optional:** Show memory service running indicator

2. **[0:12-0:15]** Type and send (SAME QUESTION):
   ```
   What tech stack am I using for my current project?
   ```

3. **[0:15-0:22]** Claude responds (expected):
   ```
   Based on our previous work, you're building a Next.js application
   with the App Router, using Prisma for the ORM, tRPC for type-safe
   APIs, TypeScript, and Tailwind CSS for styling. You're currently
   focused on implementing authentication with NextAuth.js.
   ```

4. **[0:22]** **Visual indicator:** Green ✓ or "Context Remembered" overlay

5. **[0:22-0:25]** Type follow-up:
   ```
   Add OAuth login with Google
   ```

6. **[0:25-0:30]** Claude responds (expected):
   ```
   I'll integrate Google OAuth with your existing NextAuth.js setup
   in your Next.js App Router. Let me add the Google provider...
   ```

7. **[0:30]** **End frame:** Show MCP Memory Service logo + "Never explain twice"

---

## 🎨 Visual Enhancements (Optional)

Add these in post-processing or during recording:

1. **Text overlays:**
   - "WITHOUT Memory" (Scene 1)
   - "WITH Memory" (Scene 2)
   - "Same question, different results"

2. **Indicators:**
   - Red X for "Context Lost"
   - Green checkmark for "Context Remembered"

3. **Speed adjustments:**
   - Speed up typing (1.5x)
   - Slow down Claude responses for readability

4. **Highlight boxes:**
   - Around key phrases: "I don't have information" vs "Based on our previous work"

---

## 📹 Recording Best Practices

### Do:
✅ Record at **1920x1080** or **1280x720** (GIF-friendly)
✅ Use **60 FPS** if possible (smoother GIF)
✅ **Zoom in** on Claude window (fill screen)
✅ Use **dark mode** (better contrast, smaller file size)
✅ Keep mouse movements **smooth and deliberate**
✅ **Pause 2 seconds** before each action

### Don't:
❌ Show your full desktop (focus on Claude window only)
❌ Move mouse erratically
❌ Record at 4K (GIF will be huge)
❌ Include audio (GIFs are silent)
❌ Exceed 30 seconds (attention span limit)

---

## 🎞️ Post-Recording Checklist

After recording, save the file as `demo-raw.mov` or `demo-raw.mp4` in the repo root, then:

1. **Trim** to exactly 30 seconds
2. **Add text overlays** (optional)
3. **Convert to optimized GIF** (I'll help with this)
4. **Upload to GitHub repo** under `docs/assets/demo.gif`
5. **Update README.md** with GIF link

---

## 🚀 Ready to Record?

### Quick Start Commands

```bash
# macOS: Start recording with 3-second countdown
sleep 3 && screencapture -v ~/Documents/GitHub/mcp-memory-service/demo-raw.mov

# Then follow the script above, staying within 30 seconds
```

**When you're done recording, let me know and I'll help convert it to an optimized GIF!**

---

## 📝 Alternative: Screen Recording Services

If you prefer professional polish, consider:

1. **Loom** (loom.com) - Easy recording + built-in GIF export
2. **ScreenFlow** (macOS) - Professional editing tools
3. **Camtasia** - Cross-platform with annotations

These can add overlays, text, and export optimized GIFs automatically.

---

**Questions?** Just ask! I'll help with any step of the process.
