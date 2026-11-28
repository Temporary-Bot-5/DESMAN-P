# DESMAN(P): guia para participantes no teste

Obrigado por ajudar. Aqui você encontra como instalar o DESMAN, o que experimentar e como relatar
problemas. Comece pelas seções 1–3 e consulte as demais quando precisar.

## 0. Leia primeiro

- **Este é um protótipo.** Pode fechar de repente, travar, dar respostas estranhas ou perder
  trabalho inacabado. Não confie a ele nada importante nem tarefas que não possam ser repetidas.
  Os problemas encontrados são o principal resultado do teste. Relate todos (seção 6).
- **Suas chaves ficam com você.** A chave de acesso ao modelo (chave de API) fica em um arquivo no
  seu computador e só você pode vê-la. Ela não vai para a rede DESMAN, para o proprietário, para
  modelos nem para o histórico da conversa. A rede recebe apenas uma *impressão digital*, um
  código curto que distingue a chave sem permitir recuperá-la.
- **Só você ativa a rede.** Até você clicar em **Aderir** (seção 5), o DESMAN não contata o
  servidor do proprietário. Suas tarefas, conversas e arquivos ficam no seu computador. Os pedidos
  vão só aos provedores dos modelos cujas chaves você adicionou.
- **O que o operador (dono do servidor) vê se você entrar na rede:**
  - seu nó: a impressão digital da chave do nó, a última vez que se conectou, os modelos
    emprestados e suas cotas diárias (provedor, modelo e impressão digital da chave, nunca a
    chave);
  - o trabalho que seu computador fez para a rede e o resultado;
  - os relatos de problemas que você enviou, com o texto exato da janela de confirmação (seção 6);
  - como qualquer site, o servidor vê o endereço IP de onde veio a conexão.
- **Executar código no Windows.** Instale e inicie o Docker Desktop no modo de contêineres Linux
  e execute `desman sandbox setup --yes` para autorizar o download da imagem. Você pode habilitar
  a permissão para executar código (`shell_execution`) nas configurações do DESMAN. Sem um
  ambiente isolado, o DESMAN pode escrever código e pedirá ajuda para executá-lo.
  Para instalar o Docker Desktop, use a seção Docker do assistente de configuração ou execute `desman docker install --yes`.
  Sem Docker, `desman sandbox setup --yes` configura a sandbox WSL no Windows se o WSL2 estiver disponível, ou WebAssembly para Python e pytest nos demais casos.

## 1. Instalação

O proprietário enviará o link da página com os botões de instalação. Você precisa de acesso à
internet. Não precisa de permissões de administrador.

### Windows

1. Clique em **Windows** na página para baixar `install.bat`. Se o navegador perguntar se deseja
   manter o arquivo, escolha “Manter”.
2. Abra a pasta Downloads e clique duas vezes em `install.bat`. Se aparecer a janela azul “O
   Windows protegeu o computador”, clique em “Mais informações” → “Executar assim mesmo”.
3. Uma janela preta abre. O instalador pede autorização para instalar o que falta (uv, Python e,
   se necessário, Git). Digite `y` e pressione Enter. Isso leva alguns minutos.
4. O instalador inicia o `desman` no terminal. O atalho **DESMAN** na área de trabalho também o
   abre em um terminal.

### Linux

1. Clique em **Linux** na página e copie o comando que aparecer.
2. Abra um terminal, cole o comando (Ctrl+Shift+V) e pressione Enter.
3. Se o Git estiver faltando, o instalador mostra o comando para instalá-lo. Execute-o e repita o
   passo 2.
4. Responda `y` às perguntas. O instalador inicia o `desman` no terminal. **DESMAN** aparece no
   menu de aplicativos e na área de trabalho. Autorize a abertura do atalho se solicitado.

Veja mais detalhes em [install.md](install.md).

## 2. Primeira abertura e chave de acesso

A configuração inicial é feita no terminal. Na última pergunta você pode escolher a interface
experimental no navegador. Depois, altere essa escolha com `desman interface terminal` ou
`desman interface gui`.

O DESMAN usa modelos pela internet. Você precisa de uma chave gratuita, que funciona como senha
para o programa.

1. Em **Idioma**, escolha “Português” ou seu idioma.
2. **Chave.** É mais fácil começar por um destes serviços. Ambos são gratuitos e só exigem uma
   conta:
   - **Groq** — https://console.groq.com/keys (entre na conta, selecione “Create API Key” e copie
     a chave).
   - **Google AI Studio** — https://aistudio.google.com/apikey (entre com a conta Google,
     selecione “Create API key” e copie a chave).

   Cole a chave no campo correspondente. É normal que os caracteres fiquem ocultos. Você pode
   adicionar chaves de vários provedores. Quando o limite de uma acaba, o DESMAN passa a usar
   outra. **Nunca cole uma chave na conversa**. Use apenas o campo da chave.
3. **Permissões.** Todas começam desativadas. Para o teste, você pode ativar
   “usem a rede além dos provedores de modelos” para que os agentes possam ler páginas públicas.
   Deixe as outras desativadas, a menos que o proprietário peça o contrário.
4. **desman-global.** Você pode escolher “agora não” e entrar depois (seção 5).

Os limites gratuitos são baixos e podem mudar. Se o DESMAN disser que um limite acabou, o trabalho
seguirá quando ele for renovado, geralmente em um minuto, às vezes em um dia. Se não houver
resposta por mais de uma hora, relate o problema (seção 6).

## 3. Como usar

Na interface experimental do navegador, a página é organizada assim:

- **À esquerda**, as conversas e, embaixo, **Configurações**, **Relatar um problema** e
  **Mensagens**.
- **No centro**, a conversa. Escreva com suas palavras o que precisa.
- **À direita**, **Tarefas** (o que o DESMAN está fazendo) e **Módulos** (quais modelos estão
  trabalhando).

O DESMAN responde na hora a uma pergunta simples. Um pedido maior (“escreva…”, “faça…”,
“descubra…”) vira uma *tarefa*. O DESMAN planeja, executa e confere os passos e conta o resultado
na conversa. Os arquivos ficam na pasta da tarefa, e o DESMAN mostra onde ela está.

Se um passo falhar, o DESMAN pergunta o que fazer. Aparecem
**Tentar novamente com esta resposta**, **Aceitar como está** e **Marcar como falha** abaixo da
conversa. Você também pode explicar o que corrigir.

Mudanças no DESMAN (configurações, permissões e chaves) só acontecem depois que você clicar em
“Sim” em uma janela separada. Se ela aparecer de surpresa, clique em “Não”.

Digite `/` no campo da conversa para ver os comandos. `/help` abre a ajuda e `/new` inicia uma
nova conversa. Você também pode perguntar diretamente ao DESMAN como fazer algo.

## 4. O que experimentar

Use o programa como quiser. Para começar:

1. Faça uma pergunta simples, como “o que você sabe fazer?” ou “como configuro você?”.
2. Peça um texto, como “escreva uma carta aos meus vizinhos sobre a reunião de sábado” ou “planeje
   uma viagem de dois dias a Salvador”.
3. Peça código, como “escreva um programa em Python que conte as palavras de um texto”. No Linux,
   com a permissão “executem comandos de shell” ativada, o DESMAN também o executa.
4. Continue uma tarefa com “agora acrescente…”.
5. Feche a janela no meio de uma tarefa e abra de novo pelo atalho. A tarefa deve continuar.
6. Peça uma mudança com suas palavras, como “dê respostas curtas”. Deve aparecer uma janela de
   confirmação.
7. Entre na rede (seção 5) e escreva para outro participante (seção 6).

Anote ou lembre-se de tudo que parecer estranho, confuso ou lento. Isso também é um problema.

## 5. A rede desman-global

Use `desman global donate searxng --calls N` para emprestar pesquisas do SearXNG gerenciado e `desman global search on|off` para ativar ou desativar o empréstimo se você não tiver um buscador web.

A desman-global liga os computadores dos participantes. Seu computador pode fazer pequenas partes
de tarefas alheias com os modelos que você *emprestou*. Para tarefas grandes, a rede reúne
computadores livres em grupos. Suas tarefas sempre têm prioridade. O trabalho de outras pessoas só
roda enquanto o seu DESMAN está livre.

**Para entrar:**

1. **Configurações** → **desman-global**.
2. Em **Registo**, cole o endereço do servidor que o proprietário enviará. Começa com `https://`.
3. Clique em **Aderir**. Aparece a impressão digital do seu nó.

**Para emprestar um modelo** (opcional), escolha o modelo na mesma seção, defina
**Chamadas por dia** e **Tokens por dia** e clique em **Emprestar**. Use **Retirar** para recolher
o modelo. Por enquanto, só é possível emprestar um modelo *local* (Ollama, no seu computador). Os
termos das chaves gratuitas na nuvem proíbem usá-las para trabalhar para outras pessoas. As chaves
ficam com você. Seu DESMAN faz a chamada e só envia o resultado à rede.

**Para sair**, clique em **Sair** na mesma seção. A rede para de receber dados do seu computador
na hora.

A rede funciona enquanto a página do DESMAN está aberta. Se o servidor ficar indisponível, o
DESMAN segue por conta própria e tenta de novo mais tarde.

## 6. Como relatar um problema

Se você descrever um problema com o DESMAN na conversa, ele vai sugerir um relato para você revisar.

É preciso estar na rede para enviar um relato (seção 5).

1. Clique em **Relatar um problema** no canto inferior esquerdo.
2. Descreva o que fez, o que esperava e o que aconteceu. Por exemplo, “pedi uma carta e a tarefa
   está parada há 20 minutos no passo 2 de 3”.
3. A janela mostra o **texto exato** que o operador receberá, com a descrição e as versões do
   programa e do sistema. Leia e clique em “Sim”.

As chaves nunca vão no relato. Se o DESMAN encontrar uma chave sua no texto, cancela o envio.
Conversas, perfil e arquivos da tarefa também não são enviados. Para anexar o histórico de uma
tarefa, digite `/global report --task NÚMERO "descrição"`. A janela mostra exatamente o que será
acrescentado.

Se não conseguir enviar porque a rede não conecta ou o programa não abre, escreva ao proprietário
pelo meio habitual e anexe uma captura de tela se puder.

## 7. Atualizações

As atualizações só são instaladas com sua autorização. Quando o proprietário avisar que saiu uma
versão, conecte-se à rede (seção 5). Use `/global update --show` para ver a versão e
`/global update --apply NÚMERO` para instalar. O DESMAN mostra as mudanças e pede confirmação.
Depois, feche e reabra o DESMAN. Se aparecer um aviso de “arquivos protegidos”, digite
`/protected approve` e confirme para aceitar a nova versão das regras. Para voltar à anterior, use
`/global update --revert`.

## 8. Desinstalação

- **Windows:** feche o DESMAN, pressione Win+R, cole
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` e pressione Enter.
- **Linux:** feche o DESMAN e execute
  `sh ~/.local/share/desman-app/install/install.sh --uninstall` no terminal.

No terminal, `desman uninstall` mostra o mesmo plano de remoção e `desman uninstall --yes` o
confirma. Você também pode usar Configurações → Geral → Remover DESMAN.

O programa é removido e as configurações, chaves e histórico ficam guardados caso você o instale
de novo. Para apagar tudo, acrescente `-Purge` no Windows ou `--purge` no Linux. O DESMAN mostra o
que será apagado e pede confirmação mais uma vez.

Você pode revogar as chaves que criou nos sites dos provedores (mesmos links da seção 2).
