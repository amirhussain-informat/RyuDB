import { forwardRef, useImperativeHandle, useRef } from "react";
import Editor, { type OnMount } from "@monaco-editor/react";
import type { editor as MonacoEditor } from "monaco-editor";
import type { ErrorPosition } from "../lib/types";

export interface EditorHandle {
  setParseError: (pos: ErrorPosition | null, message?: string) => void;
  insert: (text: string) => void;
  focus: () => void;
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
}

const PARSE_OWNER = "ryudb-parse";

/** Monaco-based SQL editor. Ctrl/Cmd+Enter runs the current statement; a parse
 * error from the server is underlined as a squiggle at the reported position. */
const SqlEditor = forwardRef<EditorHandle, Props>(function SqlEditor(
  { value, onChange, onRun },
  ref,
) {
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof import("monaco-editor") | null>(null);
  const runRef = useRef(onRun);
  runRef.current = onRun;

  const onMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      runRef.current();
    });
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
      theme="vs-dark"
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