// src/stream.ts
export interface StreamEvent {
  type: 'token' | 'sources' | 'status' | 'done';
  content?: string | any[];
  is_valid?: boolean;
  reason?: string;
}

export async function readStream(
  response: Response, 
  onEvent: (event: StreamEvent) => void
) {
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  
  if (!reader) return;

  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    buffer += chunk;

    const lines = buffer.split('\n\n');
    buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const jsonStr = line.slice(6);
          const data = JSON.parse(jsonStr);
          onEvent(data);
        } catch (e) {
          console.error("Failed to parse SSE JSON:", line);
        }
      }
    }
  }
}