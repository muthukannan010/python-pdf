const jsonStr = JSON.stringify([{text: "Candidate's"}]);
console.log('Original JSON:', jsonStr);

function escapeHtml(str) {
  // simulate textNode + innerHTML
  let inner = str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return inner.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const escaped = escapeHtml(jsonStr);
console.log('Escaped HTML:', escaped);

// simulate browser parsing
const unescaped = escaped.replace(/&quot;/g, '"').replace(/&#39;/g, "'");
console.log('Browser dataset:', unescaped);

const encoded = encodeURIComponent(unescaped);
console.log('Encoded:', encoded);

const decoded = decodeURIComponent(encoded);
console.log('Decoded:', decoded);

const parsed = JSON.parse(decoded);
console.log('Parsed JSON:', parsed);
