# DESMAN(P): a guide for test participants

Thank you for agreeing to help. This guide explains how to install DESMAN, what to try and how
to report a problem. You don't have to read it all at once: start with sections 1–3 and read
the rest when you need it.

## 0. Read this first

- **This is a prototype.** It may crash, hang, give strange answers and lose unfinished work.
  Don't trust it with anything important, and don't give it tasks that can't be repeated.
  The problems you find are the main result of the test: please report every one of them
  (section 7).
- **Your keys stay with you.** A model access key (an API key) is kept only in a file on your
  computer, and only you can see it. It never goes to the DESMAN network, to the owner, to a
  model or into the chat history. Only the key's *fingerprint* goes to the network: a short
  string from which the key can't be recovered, but which tells one key from another.
- **Only you turn the network on.** Until you press **Join** (section 5), DESMAN doesn't
  contact the owner's server at all. Your tasks, chats and files always stay on your
  computer; requests go only to the providers of the models whose keys you added.
- **What the operator (the server's owner) sees if you join the network:**
  - your node: the fingerprint of the node's key, when it last got in touch, which models you
    lent to the network and with what daily quota (provider, model, key fingerprint — never
    the key itself);
  - the work your computer did for the network, and its result;
  - the problem reports you sent — exactly the text that was in the confirmation window
    (section 7);
  - in the messenger: your name, and who writes to whom and when. Messages are encrypted and
    the operator can't read them — **except when there is a complaint**: if you complain
    about a message, its text goes to the operator;
  - like any website, the server sees the address it is contacted from (IP).
- **Running code on Windows.** Install and start Docker Desktop in Linux containers mode, then
  run `desman sandbox setup --yes` to authorize the image download. You can enable permission
  to run code (`shell_execution`) in DESMAN's settings. Without a sandbox, DESMAN can write
  code and will ask for help running it.
  To install Docker Desktop, use the Docker section of the setup wizard or run `desman docker install --yes`.
  Without Docker, `desman sandbox setup --yes` sets up WSL on Windows when WSL2 is available, or WebAssembly for Python and pytest otherwise.

## 1. Installation

The owner will send you a link to the page with the install buttons. You need internet
access; you don't need administrator rights.

### Windows

1. On the page, press **Windows**: the file `install.bat` is downloaded. If the browser asks
   whether to keep the file, answer "Keep".
2. Open your Downloads folder and double-click `install.bat`. If Windows shows a blue
   "Windows protected your PC" window, click "More info" → "Run anyway".
3. A black window opens. The installer asks permission to install what it needs (uv, Python
   and, if it is missing, Git): answer `y` and press Enter. This takes a few minutes.
4. The installer starts `desman` in the terminal. The **DESMAN** desktop shortcut opens it in a
   terminal too.

### Linux

1. On the page, press **Linux** and copy the command shown.
2. Open a terminal, paste the command (Ctrl+Shift+V) and press Enter.
3. If Git is missing, the installer prints the command that installs it: run it and repeat
   step 2.
4. Answer `y` to the questions. The installer starts `desman` in the terminal. **DESMAN**
   appears in the applications menu and on the desktop. Allow the shortcut to launch if
   prompted.

More details for the curious are in [install.md](install.md).

## 2. First start and a key

First-time setup runs in the terminal. Its last question lets you choose the experimental
browser interface. You can change your choice later with `desman interface terminal` or `desman interface gui`.

DESMAN works through models on the internet. They need a free key, a kind of password for a
program.

1. **Language**: choose "English" (or your own language).
2. **Key.** The easiest is to start with one of these (both are free, you only need an
   account):
   - **Groq** — https://console.groq.com/keys (sign in, "Create API Key", copy it);
   - **Google AI Studio** — https://aistudio.google.com/apikey (sign in with a Google
     account, "Create API key", copy it).

   Paste the key into the setup's field (the characters are hidden — that's expected). You
   can add several keys from different providers: when one runs out of its limit, DESMAN
   moves to another. **Never paste a key into a chat message** — only into the key field.
3. **Permissions.** Everything is off by default, and that's fine. For the test you may turn
   on "use the network beyond model providers" (agents can then read public web pages).
   Leave the rest off unless the owner asks otherwise.
4. **desman-global.** You can answer “not now” and join later (section 5).

Free limits are small and they change. If DESMAN says a limit is used up, nothing is broken:
the work continues when the limit comes back (usually in a minute, sometimes in a day). If
there is no answer for more than an hour, report it (section 7).

## 3. How to use it

In the experimental browser interface, the page is organized like this:

- **on the left**: the chats, and at the bottom the buttons **Settings**, **Report a
  problem** and **Messenger**;
- **in the middle**: the chat — write what you need in ordinary words;
- **on the right**: **Tasks** (what DESMAN is doing now) and **Modules** (which models are
  working).

DESMAN answers a simple question at once. A bigger request ("write…", "make…", "figure
out…") becomes a *task*: DESMAN plans the steps, carries them out, checks them and then tells
you in the chat what came of it. A task's files are in its own folder, and DESMAN shows you
where.

If DESMAN fails at a step, it doesn't drop it silently but asks you: the buttons **Retry with
this answer**, **Accept as is** and **Fail** appear under the chat — or answer in words what
to fix.

Anything that changes DESMAN itself (settings, permissions, keys) happens only after you
press "Yes" in a separate window. If such a window appears unexpectedly, press "No".

Commands: type `/` in the chat's field and a list appears. `/help` is the help, `/new` a new
chat. You can also simply ask DESMAN in words how to do something.

## 4. What to try

You don't need to do anything special — use it as you like. To start with:

1. Ask something simple ("what can you do?", "how do I set you up?").
2. Give it a writing task: "write a letter to my neighbours about the meeting on Saturday",
   "plan a two-day trip to Edinburgh".
3. Give it a coding task: "write a Python program that counts the words in a text". On Linux,
   with the "run shell commands" permission on, DESMAN also runs it.
4. Continue a task: "now add … to it".
5. Close the window in the middle of a task and open it again with the shortcut: the task
   should go on.
6. Ask in words to change a setting ("keep your answers short"): a confirmation window
   should appear.
7. Join the network (section 5) and write to another participant (section 6).

Remember or write down anything that seemed strange, unclear or slow: that's a problem too.

## 5. The desman-global network

Use `desman global donate searxng --calls N` to lend managed SearXNG searches. If you have no whole-web engine, use `desman global search on|off` to control borrowing.

desman-global is a network of the participants' computers. Through it DESMANs help one
another: your computer can do small parts of other people's tasks with the models you *lent*,
and the network itself gathers free computers into "clusters" for big tasks. Your own tasks
always come first: other people's work runs only while your DESMAN is idle.

**To join:**

1. **Settings** → **desman-global**.
2. In the **Registry** field, paste the server's address — the owner will send it (it starts
   with `https://`).
3. Press **Join**. Your node's fingerprint appears.

**To lend a model to the network** (optional): in the same section choose a model, set
**Calls per day** and **Tokens per day**, and press **Lend**; **Take back** returns it.
For now only a *local* model can be lent to the network (Ollama, running on your computer).
Free cloud keys can't be lent: their terms forbid working for other people. Keys never leave
your computer anyway — your DESMAN makes the call itself and gives the network only the
result.

**To leave:** the **Leave** button in the same section. The network stops getting anything
from your computer at once.

The network works while the DESMAN page is open. If the server can't be reached, DESMAN
simply works on its own and tries again later.

## 6. The messenger

Network participants can write to one another. Messages are encrypted: the operator sees who
writes to whom and when, but not the text.

1. **Name.** In the chat type `/global name your_name` (Latin letters, digits, `_`, `-`,
   3–32 characters, lower case). Others will see it.
2. **Contact.** You can write only to someone who has agreed to it too. Send a request:
   `/global contact add name`; the other person accepts with the same command and your name.
3. **Write:** `/global send name "the message"` — put the text in straight quotes `"` if it
   has more than one word (the same in all the commands below).
4. **Read:** the **Messenger** button on the left (or `/messages`, `/messages name`).
5. **Block:** `/global block name` (the other person isn't told); `/global unblock name`
   lifts it.
6. **Complain:** `/global complain name "what happened"`. To show the operator a particular
   message, add `--message NUMBER` (the number is shown when you read messages) — **the text
   of that message goes to the operator**.

DESMAN never carries out other people's messages as tasks: links and files in them don't
start anything. If you lose the node's key file (for example, when you remove the program
completely), the messages are lost.

## 7. How to report a problem

A report needs the network connection (section 5).

1. Press **Report a problem** at the bottom left.
2. Describe what you did, what you expected and what happened. For example: "I asked for a
   letter, the task has been stuck for 20 minutes at step 2 of 3".
3. A window opens with the **exact text** that goes to the operator: your description and
   the versions of the program and the system. Read it and press "Yes".

Keys never get into a report: if DESMAN finds one of your keys in the text, nothing is sent.
Chats, the profile and task files aren't sent either. To attach the history of a particular
task, type `/global report "description" --task NUMBER` — the window shows exactly what will
be added.

If you can't send it (the network doesn't connect, the program doesn't open), write to the
owner as usual and, if you can, attach a screenshot.

## 8. Updates

Updates are never installed by themselves. When the owner writes that a new version is out
(this needs the network connection, section 5): to see it, `/global update --show`; to
install it, `/global update --apply NUMBER` (DESMAN shows what will change and asks you to
confirm). After installing, close DESMAN and open it again. If after an update DESMAN talks
about "protected files", type `/protected approve` and confirm in the window — that's how
you accept the new version of the rules. To roll back: `/global update --revert`.

## 9. Removal

- **Windows:** close DESMAN, press Win+R, paste
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` and press Enter.
- **Linux:** close DESMAN and run in a terminal
  `sh ~/.local/share/desman-app/install/install.sh --uninstall`.

The program is removed, but the settings (with the keys) and the history are kept in case you
install it again. To erase everything, keys and history included, add `-Purge` (Windows) or
`--purge` (Linux) — DESMAN shows what exactly it will remove and asks once more.

You can revoke the keys you created at the providers on their websites (the same links as in
section 2).
