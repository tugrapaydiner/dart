from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from dart.runtime.server import _DEMO_HTML

NODE = shutil.which("node")
SCRIPT = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = JSON.parse(fs.readFileSync(0, 'utf8')).html;
const script = html.split('<script>')[1].split('</script>')[0];
const ids = ['run-button', 'badge', 'p', 'n', 'l', 'r', 'lm', 'rm'];
const elements = Object.fromEntries(ids.map(id => [id, {textContent:'stale', className:'', value:'', disabled:false}]));
elements.p.value = 'example prompt';
elements.n.value = '12';
const context = vm.createContext({document:{getElementById:id => elements[id]}});
vm.runInContext(script, context);
const hash = 'a'.repeat(64);
const response = {identical:true, baseline:{text:'answer', wall_ms:13, tokens:2, output_hash:hash},
                  dart:{text:'answer', wall_ms:23, tokens:2, output_hash:hash, acceptance_rate:0.5, avg_accept_run:1}};
async function main() {
    for (const [fetch, expected] of [
        [async () => ({ok:false, json:async () => ({error:'context too long <b>'})}), 'context too long <b>'],
        [async () => ({ok:false, json:async () => ({})}), 'Request failed'],
        [async () => {throw new Error('offline');}, 'offline'],
        [async () => ({ok:true, json:async () => {throw new Error('bad JSON');}}), 'bad JSON'],
    ]) {
        context.fetch = fetch;
        await context.run();
        assert.equal(elements['run-button'].disabled, false);
        assert.equal(elements.badge.textContent, expected);
        assert.equal(elements.badge.className, 'bad');
        for (const id of ['l','r','lm','rm']) assert.equal(elements[id].textContent, '');
    }
    for (const id of ids) assert(html.includes('id="' + id + '"'));
    let release;
    context.fetch = (url, options) => {
        assert.equal(url, '/v1/dart_demo');
        assert.deepEqual(JSON.parse(options.body), {prompt:'example prompt', max_tokens:12});
        return new Promise(resolve => { release = resolve; });
    };
    const pending = context.run();
    assert.equal(elements['run-button'].disabled, true);
    assert.equal(elements.badge.textContent, 'running...');
    for (const id of ['l','r','lm','rm']) assert.equal(elements[id].textContent, '');
    release({ok:true, json:async () => response});
    await pending;
    assert.equal(elements['run-button'].disabled, false);
    assert.equal(elements.badge.textContent, 'MATCHING TOKENS');
    assert.equal(elements.badge.className, 'ok');
    assert.equal(elements.l.textContent, 'answer');
    assert.equal(elements.r.textContent, 'answer');
    assert(elements.lm.textContent.includes(hash) && elements.rm.textContent.includes(hash));
    assert(elements.lm.textContent.includes('13 ms') && elements.rm.textContent.includes('23 ms'));
    assert(elements.rm.textContent.includes('accept 0.5'));
    context.fetch = async () => ({ok:true, json:async () => ({...response, identical:false})});
    await context.run();
    assert.equal(elements.badge.textContent, 'OUTPUT MISMATCH');
    assert.equal(elements.badge.className, 'bad');
    process.stdout.write(JSON.stringify({passed:6}));
}
main().catch(error => {process.stderr.write(error.stack); process.exitCode=1;});
"""


@pytest.mark.skipif(NODE is None, reason="demo script checks need Node.js")
def test_demo_page_handles_responses_and_displays_complete_hashes(tmp_path):
    result = subprocess.run([NODE, "-e", SCRIPT], input=json.dumps({"html": _DEMO_HTML}),
                            capture_output=True, text=True, cwd=tmp_path, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"passed": 6}
