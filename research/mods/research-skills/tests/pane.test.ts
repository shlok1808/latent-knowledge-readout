import { test, expect } from 'claude-code/testing'

test('the pane lists skill names by group', async $ => {
  for (const surface of ['terminal', 'desktop', 'mobile'] as const) {
    const ui = await $.ui.mount({
      plugin: 'research-skills',
      surface,
      component: 'Pane',
      requestId: 'research-skills',
      props: { title: 'Skills' },
      viewport: { columns: 60, rows: 20 },
    })
    expect(await ui.find({ type: 'Text', text: /transformer-lens/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /code-review/ })).toBeDefined()
    await ui.unmount()
  }
})
