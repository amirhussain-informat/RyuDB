import { forwardRef, useImperativeHandle, useRef } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import type { editor as MonacoEditor, languages as MonacoLanguages } from "monaco-editor";
import type { ErrorPosition } from "../lib/types";
import { buildSqlSuggestions, type Schema, type SuggestionKind } from "../lib/autocomplete";

export type { Schema };

export interface EditorHandle {
  setParseError: (pos: ErrorPosition | null, message?: string) => void;
  insert: (text: string) => void;
  focus: () => void;
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
  /** Catalog schema for SQL autocompletion (table names + columns). */
  schema?: Schema;
  /** Monaco theme id; switches the editor chrome between dark and light. */
  theme?: string;
}

/** Register a schema-aware SQL completion provider against the shared monaco
 *  instance. Called once from onMount; the provider reads `schemaRef.current`
 *  at completion time so it always sees the latest schema without re-registering.
 *  The pure suggestion logic lives in lib/autocomplete.ts (unit-tested); this
 *  just maps the kind strings to Monaco CompletionItemKind enums + the range. */
function registerSqlCompletion(
  monaco: typeof import("monaco-editor"),
  schemaRef: { current: Schema | undefined },
): void {
  const kindMap: Record<SuggestionKind, MonacoLanguages.CompletionItemKind> = {
    keyword: monaco.languages.CompletionItemKind.Keyword,
    table: monaco.languages.CompletionItemKind.Class,
    column: monaco.languages.CompletionItemKind.Field,
  };
  monaco.languages.registerCompletionItemProvider("sql", {
    triggerCharacters: ["."],
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const lineUntil = model.getValueInRange({
        startLineNumber: position.lineNumber,
        startColumn: 1,
        endLineNumber: position.lineNumber,
        endColumn: word.startColumn,
      });
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const suggestions: MonacoLanguages.CompletionItem[] = buildSqlSuggestions(
        lineUntil,
        schemaRef.current,
      ).map((s) => ({
        label: s.label,
        kind: kindMap[s.kind],
        insertText: s.insertText,
        detail: s.detail,
        range,
        sortText: s.sortText,
      }));
      return { suggestions };
    },
  });
}

const PARSE_OWNER = "ryudb-parse";

/** Monaco-based SQL editor. Ctrl/Cmd+Enter runs the current statement; a parse
 * error from the server is underlined as a squiggle at the reported position. */
const SqlEditor = forwardRef<EditorHandle, Props>(function SqlEditor(
  { value, onChange, onRun, schema, theme = "vs-dark" },
  ref,
) {
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import("monaco-editor") | null>(null);
  const runRef = useRef(onRun);
  runRef.current = onRun;
  // The completion provider reads this ref (registered once on mount) so it
  // sees the latest schema without re-registering on every catalog refresh.
  const schemaRef = useRef<Schema | undefined>(schema);
  schemaRef.current = schema;
  // Guards against double registration under React StrictMode (onMount can fire
  // twice in dev); registerCompletionItemProvider is language-global, so a second
  // registration would double every suggestion.
  const registeredRef = useRef(false);

  const onMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      runRef.current();
    });
    if (!registeredRef.current) {
      registeredRef.current = true;
      registerSqlCompletion(monaco, schemaRef);
    }
  };

  useImperativeHandle(ref, () => ({
    setParseError: (pos, message) => {
      const editor = editorRef.current;
      const monaco = monacoRef.current;
      if (!editor || !monaco) return;
      const model = editor.getModel();
      if (!model) return;
      if (pos === null) {
        monaco.editor.setModelMarkers(model, PARSE_OWNER, []);
        return;
      }
      // server positions are 1-based line/col; Monaco is 1-based too.
      const line = pos.line;
      const col = Math.max(1, pos.col);
      const marker: MonacoEditor.IMarkerData = {
        startLineNumber: line,
        startColumn: col,
        endLineNumber: line,
        endColumn: col + 1,
        message: message ?? "parse error",
        severity: monaco.MarkerSeverity.Error,
      };
      monaco.editor.setModelMarkers(model, PARSE_OWNER, [marker]);
      editor.revealLineInCenter(line);
    },
    insert: (text) => {
      const editor = editorRef.current;
      if (!editor) return;
      const sel = editor.getSelection();
      editor.executeEdits("ryudb-insert", [
        { range: sel ?? editor.getModel()!.getFullModelRange(), text },
      ]);
      editor.focus();
    },
    focus: () => editorRef.current?.focus(),
  }));

  return (
    <Editor
      height="100%"
      defaultLanguage="sql"
      theme={theme}
      value={value}
      onChange={(v) => onChange(v ?? "")}
      onMount={onMount}
      options={{
        fontSize: 13,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 2,
        wordWrap: "on",
      }}
    />
  );
});

export default SqlEditor;