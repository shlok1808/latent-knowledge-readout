import type { Register } from 'claude-code'

const PANE = 'candidate-directions'
const TITLE = 'Candidate Directions'

const DIRECTIONS: [string, string][] = [
  [
    '1. Bypass vs override',
    'Interpretability can show that models know things they don\'t act on, but not what happens to that knowledge at the moment of action. We characterize the geometry of the knowledge-to-action map across layers and find whether knowledge is lost, overridden, or moved into the action\'s null space.',
  ],
  [
    '2. Where shared geometry breaks',
    'Representation and behavior share geometry for simple concepts, but nobody has tested this when a model acts against its knowledge. We test whether the know-do gap is a break in that isometry, and whether steering along the manifold repairs behavior where linear steering fails.',
  ],
  [
    '3. Jump vs drift (later)',
    'Agents switch from honest to dishonest behavior over a task, but whether that switch is gradual or sudden is unknown. We treat the trajectory as a dynamical system and test for a basin crossing at the switch point.',
  ],
]

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'directions', description: 'Show candidate research directions' })
    void $.ui.open({ id: PANE, title: TITLE })

    return next(e)
  })

  on('command.run', { command: 'directions' }, async $ => {
    await $.ui.open({ id: PANE, title: TITLE })

    return { text: 'Candidate Directions pane opened.' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button } = $.ui.resolve(e)

    return (
      <Box flexDirection="column" gap={1}>
        <Text dimColor>Ideas to explore, not decisions.</Text>
        {DIRECTIONS.map(([name, text]) => (
          <Box flexDirection="column">
            <Text bold>{name}</Text>
            <Text>{text}</Text>
            <Button
              key={'copy-' + name}
              plain
              dimColor
              label="Copy"
              onPress={press => {
                void $.ui.copy({ text: name + '\n' + text, surface: press.surface })
              }}
            />
          </Box>
        ))}
      </Box>
    )
  })
}
