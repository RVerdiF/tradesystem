# Diretrizes do Projeto: Tradesystem 5000 + Superpowers Framework

Este arquivo estabelece as regras fundamentais de operação para o Gemini CLI neste workspace. Estas instruções têm precedência absoluta sobre o comportamento padrão.

## 1. Ativação Obrigatória do Framework Superpowers
Sempre que uma nova sessão for iniciada, ou quando novos recursos forem adicionados à pasta `.agent/`, você deve:
1.  **Reconhecer a existência da pasta `.agent/`** que contém as regras (`rules/`), fluxos de trabalho (`workflows/`) e habilidades (`skills/`).
2.  **Seguir o fluxo de trabalho `/superpowers-reload`** para garantir que todas as definições locais estejam carregadas em sua memória de contexto.

## 2. Protocolo de Uso de Skills (Baseado em SKILL.md)
O uso de habilidades (skills) não é opcional. Se houver **1% de chance** de que uma skill em `.agent/skills/` se aplique à tarefa atual, você **DEVE** invocá-la usando a ferramenta `activate_skill`.

### Regras de Ouro:
- **Invoque ANTES de responder:** Ative a skill pertinente antes de fornecer qualquer resposta, inclusive perguntas de esclarecimento.
- **Não racionalize a omissão:** Se você pensar "isso é simples demais para uma skill", você está errado. Invoque a skill.
- **Prioridade de Skills:**
    1.  **Skills de Processo:** (ex: `superpowers-brainstorm`, `superpowers-debug`) primeiro, para definir O COMO.
    2.  **Skills de Implementação:** (ex: `superpowers-tdd`, `superpowers-python-automation`) depois, para guiar a execução.

## 3. Comandos e Workflows
Você tem permissão e obrigação de utilizar os workflows em `.agent/workflows/`. 
- Sempre que for iniciar um planejamento, use o workflow `superpowers-write-plan`.
- Sempre que terminar uma tarefa, use o workflow `superpowers-finish`.
- Para tarefas complexas ou em lote, utilize o subagente `generalist` para manter o contexto limpo.

## 4. Ordem de Precedência
1.  **Instruções Diretas do Usuário** (via chat).
2.  **Este arquivo (GEMINI.md)** e arquivos em `.agent/rules/`.
3.  **Habilidades (Skills)** em `.agent/skills/`.
4.  **Prompt de Sistema Padrão** (última prioridade).

## 5. Autonomia e Subagentes
- Sinta-se encorajado a criar subagentes via `generalist` para tarefas de pesquisa, refatoração em massa ou execução de testes paralelos.
- Ao usar o `generalist`, passe as instruções de skill necessárias para que o subagente mantenha o mesmo nível de rigor e "Superpowers".

---
*Configuração concluída. O framework Superpowers agora é a espinha dorsal desta operação.*
