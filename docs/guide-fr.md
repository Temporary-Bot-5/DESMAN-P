# DESMAN(P): guide des participants au test

Merci de votre aide. Ce guide explique comment installer DESMAN, quoi essayer et comment signaler
un problème. Commencez par les sections 1 à 3 et consultez le reste au besoin.

## 0. À lire en premier

- **C’est un prototype.** Il peut se fermer brusquement, se bloquer, donner des réponses étranges
  ou perdre du travail en cours. Ne lui confiez rien d’important ni de tâches impossibles à
  répéter. Les problèmes trouvés sont le principal résultat du test. Signalez-les tous (section
  7).
- **Vos clés restent chez vous.** Une clé d’accès à un modèle (clé API) est stockée dans un
  fichier sur votre ordinateur et vous seul pouvez la voir. Elle n’est envoyée ni au réseau
  DESMAN, ni au propriétaire, ni à un modèle. Elle ne figure pas dans l’historique du chat. Le
  réseau reçoit seulement son *empreinte*, une courte chaîne qui distingue la clé des autres sans
  permettre de la retrouver.
- **Vous seul activez le réseau.** Tant que vous ne cliquez pas sur **Rejoindre** (section 5),
  DESMAN ne contacte pas le serveur du propriétaire. Vos tâches, conversations et fichiers restent
  sur votre ordinateur. Les requêtes vont seulement aux fournisseurs des modèles dont vous avez
  ajouté les clés.
- **Ce que l’opérateur (propriétaire du serveur) voit si vous rejoignez le réseau :**
  - votre nœud : l’empreinte de sa clé, son dernier contact, les modèles prêtés et leurs quotas
    quotidiens (fournisseur, modèle et empreinte de la clé, jamais la clé) ;
  - le travail effectué par votre ordinateur pour le réseau et son résultat ;
  - vos signalements de problèmes, avec le texte exact de la fenêtre de confirmation (section 7) ;
  - dans la messagerie, votre nom et qui écrit à qui et quand. Les messages sont chiffrés et
    l’opérateur ne peut pas les lire, **sauf en cas de plainte**. Si vous signalez un message, il
    en reçoit le texte ;
  - comme sur tout site web, l’adresse IP d’où vient la connexion.
- **Exécuter du code sous Windows.** Installez et lancez Docker Desktop en mode conteneurs Linux,
  puis exécutez `desman sandbox setup --yes` pour autoriser le téléchargement de l’image. Vous
  pouvez activer l’autorisation d’exécuter du code (`shell_execution`) dans les paramètres de
  DESMAN. Sans bac à sable, DESMAN peut écrire du code et vous demandera de l’aide pour l’exécuter.
  Pour installer Docker Desktop, utilisez la section Docker de l’assistant de configuration ou lancez `desman docker install --yes`.
  Sans Docker sur Windows ou macOS, exécutez `desman sandbox setup --yes` pour configurer le bac à sable WebAssembly pour Python et pytest.

## 1. Installation

Le propriétaire vous enverra le lien de la page d’installation. Il faut un accès à Internet. Les
droits d’administration ne sont pas nécessaires.

### Windows

1. Cliquez sur **Windows** pour télécharger `install.bat`. Si le navigateur vous demande si vous
   voulez garder le fichier, choisissez « Conserver ».
2. Ouvrez le dossier Téléchargements et double-cliquez sur `install.bat`. Si la fenêtre bleue «
   Windows a protégé votre ordinateur » apparaît, cliquez sur « Informations complémentaires » → «
   Exécuter quand même ».
3. Une fenêtre noire s’ouvre. Le programme demande l’autorisation d’installer uv, Python et, si
   nécessaire, Git. Tapez `y` et appuyez sur Entrée. Cela prend quelques minutes.
4. À la fin, le navigateur ouvre la page DESMAN et un raccourci **DESMAN** apparaît sur le bureau.
   Utilisez-le ensuite pour lancer le programme.

### Linux

1. Cliquez sur **Linux** et copiez la commande affichée.
2. Ouvrez un terminal, collez la commande (Ctrl+Maj+V) et appuyez sur Entrée.
3. Si Git manque, le programme affiche la commande pour l’installer. Exécutez-la et reprenez à
   l’étape 2.
4. Répondez `y` aux questions. À la fin, le navigateur s’ouvre et **DESMAN** apparaît dans le menu
   des applications et sur le bureau. Si le bureau vous demande d’autoriser le lancement du
   raccourci, acceptez.

Pour plus de détails, consultez [install.md](install.md).

## 2. Premier lancement et clé d’accès

DESMAN utilise des modèles sur Internet. Il leur faut une clé gratuite, une sorte de mot de passe
pour le programme. La configuration s’ouvre automatiquement au premier lancement.

1. Dans **Langue**, choisissez « Français » ou votre langue.
2. **Clé.** Le plus simple est de commencer par l’un de ces services. Ils sont gratuits et
   demandent seulement un compte :
   - **Groq** — https://console.groq.com/keys (connectez-vous, choisissez « Create API Key »,
     copiez la clé).
   - **Google AI Studio** — https://aistudio.google.com/apikey (connectez-vous avec votre compte
     Google, choisissez « Create API key », copiez la clé).

   Collez la clé dans le champ prévu. Il est normal que les caractères soient masqués. Vous pouvez
   ajouter des clés de plusieurs fournisseurs. Quand une clé atteint sa limite, DESMAN passe à la
   suivante. **Ne collez jamais une clé dans un message**. Utilisez uniquement le champ de la clé.
3. **Autorisations.** Tout est désactivé au départ. Pour le test, vous pouvez activer «
   utiliser le réseau au-delà des fournisseurs de modèles » afin que les agents lisent des pages
   web publiques. Laissez le reste désactivé sauf demande du propriétaire.
4. **desman-global** est la dernière question. Vous pouvez choisir « pas maintenant » et rejoindre
   le réseau plus tard (section 5).

Les limites gratuites sont basses et peuvent changer. Si DESMAN annonce qu’une limite est
atteinte, le travail reprend à son renouvellement, généralement après une minute, parfois après un
jour. Sans réponse au bout d’une heure, signalez-le (section 7).

## 3. Utilisation

DESMAN s’ouvre dans une page du navigateur :

- **À gauche**, les conversations, puis **Paramètres**, **Signaler un problème** et **Messagerie**
  en bas.
- **Au centre**, la conversation. Écrivez avec vos mots ce dont vous avez besoin.
- **À droite**, **Tâches** (ce que fait DESMAN) et **Modules** (les modèles au travail).

DESMAN répond immédiatement à une question simple. Une demande plus large (« écris… », « prépare…
», « trouve… ») devient une *tâche*. DESMAN prévoit les étapes, les exécute, les vérifie et donne
le résultat dans la conversation. Il indique où se trouve le dossier des fichiers de la tâche.

Si une étape échoue, DESMAN vous demande quoi faire. Les boutons **Réessayer avec cette réponse**,
**Accepter tel quel** et **Marquer en échec** s’affichent sous la conversation. Vous pouvez aussi
expliquer quoi corriger.

Les changements dans DESMAN (paramètres, autorisations, clés) nécessitent un clic sur « Oui » dans
une fenêtre à part. Si elle s’affiche sans que vous vous y attendiez, cliquez sur « Non ».

Tapez `/` dans le champ de saisie pour voir les commandes. `/help` ouvre l’aide et `/new` lance
une nouvelle conversation. Vous pouvez aussi demander directement à DESMAN comment faire.

## 4. Quoi essayer

Utilisez DESMAN comme vous voulez. Pour commencer :

1. Posez une question simple, comme « que sais-tu faire ? » ou « comment te configurer ? ».
2. Demandez un texte, par exemple « écris à mes voisins au sujet de la réunion de samedi » ou «
   prépare un séjour de deux jours à Lyon ».
3. Demandez du code : « écris un programme Python qui compte les mots dans un texte ». Sous Linux,
   DESMAN le lance aussi si l’autorisation « exécuter des commandes shell » est activée.
4. Poursuivez une tâche : « ajoute maintenant… ».
5. Fermez la fenêtre au milieu d’une tâche et rouvrez-la avec le raccourci. La tâche devrait
   reprendre.
6. Demandez un changement, par exemple « fais des réponses courtes ». Une fenêtre de confirmation
   devrait s’ouvrir.
7. Rejoignez le réseau (section 5) et écrivez à un autre participant (section 6).

Notez ou retenez tout ce qui vous a semblé étrange, confus ou lent. Cela compte aussi comme un
problème.

## 5. Le réseau desman-global

Utilisez `desman global donate searxng --calls N` pour prêter le moteur SearXNG géré et `desman global search on|off` pour activer ou désactiver l’emprunt si vous n’avez pas de moteur web.

desman-global relie les ordinateurs des participants. Votre ordinateur peut traiter de petites
parties des tâches des autres avec les modèles que vous avez *prêtés*. Pour les grandes tâches, le
réseau regroupe les ordinateurs libres. Vos tâches passent toujours en premier. Le travail des
autres se fait seulement quand votre DESMAN est libre.

**Pour rejoindre le réseau :**

1. **Paramètres** → **desman-global**.
2. Dans **Registre**, collez l’adresse du serveur que vous enverra le propriétaire. Elle commence
   par `https://`.
3. Cliquez sur **Rejoindre**. L’empreinte de votre nœud apparaît.

**Pour prêter un modèle** (facultatif), choisissez-le dans la même section, fixez
**Appels par jour** et **Tokens par jour**, puis cliquez sur **Prêter**. Utilisez **Reprendre**
pour le retirer. Seuls les modèles *locaux* sont acceptés pour l’instant (Ollama, sur votre
ordinateur). Les conditions des clés cloud gratuites interdisent de travailler pour d’autres
personnes. Vos clés restent sur votre ordinateur. DESMAN fait lui-même l’appel et transmet
seulement le résultat au réseau.

**Pour partir**, cliquez sur **Quitter** au même endroit. Le réseau cesse aussitôt de recevoir des
données de votre ordinateur.

Le réseau fonctionne tant que la page DESMAN reste ouverte. Si le serveur est inaccessible, DESMAN
continue de travailler seul et réessaie plus tard.

## 6. Messagerie

Les participants peuvent s’écrire. Les messages sont chiffrés. L’opérateur voit qui écrit à qui et
quand, sans pouvoir lire le texte.

1. **Nom.** Tapez `/global name votre_nom` dans la conversation. Utilisez 3 à 32 caractères :
   lettres latines minuscules, chiffres, `_` et `-`. Les autres verront ce nom.
2. **Contact.** Vous pouvez écrire à quelqu’un qui a aussi donné son accord. Envoyez
   `/global contact add nom`. L’autre personne accepte avec la même commande et votre nom.
3. **Écrire.** Utilisez `/global send nom "le message"`. Si le texte contient plusieurs mots,
   mettez-le entre guillemets droits `"`, y compris dans les commandes suivantes.
4. **Lire.** Cliquez sur **Messagerie** à gauche ou tapez `/messages` ou `/messages nom`.
5. **Bloquer.** Utilisez `/global block nom`. L’autre personne n’est pas avertie.
   `/global unblock nom` débloque le contact.
6. **Porter plainte.** Utilisez `/global complain nom "ce qui s’est passé"`. Pour montrer un
   message précis à l’opérateur, ajoutez `--message NUMÉRO`. Le numéro s’affiche à la lecture.
   **Le texte de ce message ira à l’opérateur**.

DESMAN ne traite jamais les messages d’autres personnes comme des tâches. Leurs liens et fichiers
ne lancent rien. Si vous perdez le fichier de la clé du nœud, par exemple en désinstallant
complètement le programme, vous perdez les messages.

## 7. Signaler un problème

Il faut être connecté au réseau (section 5).

1. Cliquez sur **Signaler un problème** en bas à gauche.
2. Décrivez ce que vous avez fait, ce que vous attendiez et ce qui s’est passé. Par exemple : «
   J’ai demandé une lettre et la tâche est bloquée depuis 20 minutes à l’étape 2 sur 3 ».
3. Une fenêtre montre le **texte exact** destiné à l’opérateur, avec votre description et les
   versions du programme et du système. Lisez-le et cliquez sur « Oui ».

Les clés ne figurent jamais dans un signalement. Si DESMAN trouve une de vos clés dans le texte,
il annule l’envoi. Les conversations, le profil et les fichiers ne sont pas envoyés non plus. Pour
joindre l’historique d’une tâche, tapez `/global report "description" --task NUMÉRO`. La fenêtre
montre exactement ce qui sera ajouté.

Si vous ne pouvez pas envoyer le signalement parce que le réseau ou le programme ne fonctionne
pas, écrivez au propriétaire par le moyen habituel et joignez une capture d’écran si possible.

## 8. Mises à jour

Les mises à jour demandent votre accord. Quand le propriétaire annonce une nouvelle version,
connectez-vous au réseau (section 5). Tapez `/global update --show` pour la voir et
`/global update --apply NUMÉRO` pour l’installer. DESMAN affiche les changements et demande
confirmation. Ensuite, fermez et rouvrez DESMAN. S’il signale des « fichiers protégés », tapez
`/protected approve` et confirmez pour accepter les nouvelles règles. Pour revenir à la version
précédente, tapez `/global update --revert`.

## 9. Désinstallation

- **Windows :** fermez DESMAN, appuyez sur Win+R, collez
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` et appuyez sur Entrée.
- **Linux :** fermez DESMAN et exécutez
  `sh ~/.local/share/desman-app/install/install.sh --uninstall` dans un terminal.

Le programme est supprimé et les paramètres, les clés et l’historique restent en cas de
réinstallation. Pour tout effacer, ajoutez `-Purge` sous Windows ou `--purge` sous Linux. DESMAN
montre ce qui sera supprimé et vous demande une nouvelle confirmation.

Vous pouvez révoquer les clés sur les sites des fournisseurs (mêmes liens qu’à la section 2).
