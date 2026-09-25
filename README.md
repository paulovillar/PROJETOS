# Projetos

## Jogo da velha online (`jogo-da-velha/`)

Jogo da velha para duas pessoas, cada uma no seu aparelho, em tempo real. Um único arquivo HTML, sem backend próprio.

### Como jogar

**Hospedado (GitHub Pages ou qualquer host estático)** — conexão peer-to-peer via WebRTC ([PeerJS](https://peerjs.com), usando o broker público gratuito só para o handshake):

1. Abra `jogo-da-velha/index.html` no navegador e clique em **Criar sala**.
2. Copie o link gerado (termina em `#codigo`) e mande para a Elisa.
3. Quando ela abrir, os dois estão conectados: quem criou joga de ✕, quem entrou joga de ◯ (dá para trocar).

Para publicar no GitHub Pages: *Settings → Pages → Deploy from a branch*, escolha a branch e a pasta raiz. O jogo fica em `https://<usuario>.github.io/<repo>/jogo-da-velha/`.

**Como Artifact no claude.ai** — a mesma página usa a capability `room` (presença em tempo real): basta os dois abrirem o mesmo link do artifact.

**Sem conexão** — se nenhum modo online estiver disponível, a página cai para o modo local (os dois jogam no mesmo aparelho).

### Como funciona

- O estado da partida (`tabuleiro`, `vez`, `placar`, `seq`) é transmitido inteiro a cada jogada; cada cliente adota o estado com maior `seq` (desempate determinístico), então os dois convergem mesmo após reconexão.
- Placar, nome e símbolo ficam salvos no `localStorage` de cada aparelho.
- Quem começa alterna a cada nova partida.
