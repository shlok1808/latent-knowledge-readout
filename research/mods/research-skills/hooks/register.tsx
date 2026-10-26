import type { Register } from 'claude-code'

const PANE = 'research-skills'
const TITLE = 'Skills'

const GROUPS: [string, string][] = [
  ['Research', 'deep-research, brainstorming-research-ideas, creative-thinking-for-research'],
  ['Design', 'transformer-lens, pyvene, nnsight, saelens'],
  ['Run', 'vllm, sglang, weights-and-biases, modal, ml-training-recipes'],
  ['Check', 'code-review, simplify'],
  ['Code', 'engineering:debug, engineering:testing-strategy, engineering:architecture'],
]

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'skills', description: 'Show skill names' })
    void $.ui.open({ id: PANE, title: TITLE })

    return next(e)
  })

  on('command.run', { command: 'skills' }, async $ => {
    await $.ui.open({ id: PANE, title: TITLE })

    return { text: 'Skills pane opened.' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)

    return (
      <Box flexDirection="column">
        {GROUPS.map(([group, names]) => (
          <Text>
            <Text bold>{group}: </Text>
            {names}
          </Text>
        ))}
      </Box>
    )
  })
}
