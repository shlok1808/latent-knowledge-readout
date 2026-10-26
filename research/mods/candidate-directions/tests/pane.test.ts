import { test, expect } from 'claude-code/testing'

test('the pane lists the three directions', async $ => {
  for (const surface of ['terminal', 'desktop', 'mobile'] as const) {
    const ui = await $.ui.mount({
      plugin: 'candidate-directions',
      surface,
      component: 'Pane',
      requestId: 'candidate-directions',
      props: { title: 'Candidate Directions' },
      viewport: { columns: 60, rows: 40 },
    })
    for (const text of [/Bypass vs override/, /shared geometry breaks/, /Jump vs drift/]) {
      expect(await ui.find({ type: 'Text', text })).toBeDefined()
    }
    await ui.unmount()
  }
})
