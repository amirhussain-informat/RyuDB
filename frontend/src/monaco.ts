// Bundle Monaco from the local `monaco-editor` package instead of loading it
// from a CDN at runtime, so the worksheet works fully offline. Imported for its
// side effects before <App> mounts (see main.tsx): it wires Monaco's web worker
// and tells @monaco-editor/react to use this bundled instance.
//
// Minimal surface: the editor API + editor contributions (find, folding,
// multi-cursor, bracket matching, ...) + the SQL tokenizer. The full
// `monaco-editor` entry (editor.main) bundles EVERY language + language service
// (~3 MB gzipped); this SQL-only subset is far smaller and all a SQL worksheet
// needs.

import * as monaco from "monaco-editor/esm/vs/editor/editor.api";
import "monaco-editor/esm/vs/editor/editor.all";
import "monaco-editor/esm/vs/basic-languages/sql/sql.contribution";
import { loader } from "@monaco-editor/react";

// Monaco runs background work (tokenization, simple language services) in a web
// worker. Vite's `?worker` suffix turns the worker entry into a constructable
// Worker class. SQL has no dedicated language-service worker in core Monaco, so
// the editor worker is the only one we register (and the fallback for any label).
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";

(self as unknown as { MonacoEnvironment: monaco.Environment }).MonacoEnvironment = {
  getWorker() {
    return new EditorWorker();
  },
};

// Use this bundled Monaco instead of fetching one from a CDN.
loader.config({ monaco });

export { monaco };