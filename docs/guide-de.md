# DESMAN(P): Leitfaden für Testteilnehmende

Danke, dass Sie mithelfen. Hier erfahren Sie, wie Sie DESMAN installieren, was Sie ausprobieren
können und wie Sie Probleme melden. Fangen Sie mit den Abschnitten 1–3 an und lesen Sie den Rest
bei Bedarf.

## 0. Bitte zuerst lesen

- **Das ist ein Prototyp.** Das Programm kann abstürzen, hängen bleiben, seltsame Antworten geben
  und angefangene Arbeit verlieren. Vertrauen Sie ihm nichts Wichtiges an und geben Sie ihm keine
  Aufgaben, die Sie nicht wiederholen können. Die gefundenen Probleme sind das wichtigste
  Testergebnis. Melden Sie bitte jedes davon (Abschnitt 7).
- **Ihre Schlüssel bleiben bei Ihnen.** Ein Zugriffsschlüssel für ein Modell (API-Schlüssel) liegt
  nur in einer Datei auf Ihrem Rechner, die nur Sie einsehen können. Der Schlüssel gelangt weder
  ins DESMAN-Netzwerk noch zum Betreiber, zu einem Modell oder in den Chatverlauf. Das Netzwerk
  erhält nur den *Fingerabdruck*: eine kurze Zeichenfolge, die den Schlüssel von anderen
  unterscheidet und aus der er sich nicht wiederherstellen lässt.
- **Nur Sie schalten das Netzwerk ein.** Erst wenn Sie auf **Beitreten** klicken (Abschnitt 5),
  nimmt DESMAN Kontakt zum Server des Betreibers auf. Ihre Aufgaben, Chats und Dateien bleiben auf
  Ihrem Rechner. Anfragen gehen nur an die Anbieter der Modelle, deren Schlüssel Sie hinzugefügt
  haben.
- **Was der Betreiber (Eigentümer des Servers) nach Ihrem Beitritt sieht:**
  - Ihren Knoten: den Fingerabdruck seines Schlüssels, den letzten Kontakt, die verliehenen
    Modelle und deren Tageslimits (Anbieter, Modell und Fingerabdruck, niemals den Schlüssel
    selbst);
  - die Arbeit Ihres Rechners für das Netzwerk und deren Ergebnis;
  - Ihre abgeschickten Problemmeldungen mit genau dem Text aus dem Bestätigungsfenster (Abschnitt
    7);
  - im Nachrichtendienst Ihren Namen und wer wem wann schreibt. Nachrichten sind verschlüsselt und
    der Betreiber kann sie nicht lesen, **außer bei einer Beschwerde**. Wenn Sie eine Nachricht
    melden, erhält er deren Text;
  - wie bei jeder Website die IP-Adresse, von der aus der Server kontaktiert wird.
- **Code unter Windows ausführen.** Installieren und starten Sie Docker Desktop im Modus für
  Linux-Container. Führen Sie dann `desman sandbox setup --yes` aus, um den Download des Images
  zu bestätigen. In den DESMAN-Einstellungen können Sie die Berechtigung zur Codeausführung
  (`shell_execution`) aktivieren. Ohne Sandbox kann DESMAN Code schreiben und bittet beim
  Ausführen um Hilfe.
  Zur Installation von Docker Desktop nutzen Sie den Abschnitt „Docker“ im Einrichtungsassistenten oder führen Sie `desman docker install --yes` aus.
  Ohne Docker richtet `desman sandbox setup --yes` unter Windows mit WSL2 die WSL-Sandbox ein, ansonsten WebAssembly für Python und pytest.

## 1. Installation

Der Betreiber schickt Ihnen den Link zur Seite mit den Installationsschaltflächen. Sie brauchen
Internetzugang. Administratorrechte sind nicht nötig.

### Windows

1. Klicken Sie auf der Seite auf **Windows**. Dadurch wird `install.bat` heruntergeladen. Wenn der
   Browser nachfragt, wählen Sie „Beibehalten“.
2. Öffnen Sie den Ordner Downloads und doppelklicken Sie auf `install.bat`. Wenn das blaue Fenster
   „Der Computer wurde durch Windows geschützt“ erscheint, klicken Sie auf „Weitere Informationen“
   → „Trotzdem ausführen“.
3. Ein schwarzes Fenster öffnet sich. Das Installationsprogramm fragt, ob es uv, Python und bei
   Bedarf Git installieren darf. Geben Sie `y` ein und drücken Sie die Eingabetaste. Das
   dauert ein paar Minuten.
4. Das Installationsprogramm startet `desman` im Terminal. Die Desktop-Verknüpfung **DESMAN**
   startet das Programm ebenfalls in einem Terminal.

### Linux

1. Klicken Sie auf der Seite auf **Linux** und kopieren Sie den angezeigten Befehl.
2. Öffnen Sie ein Terminal, fügen Sie den Befehl ein (Strg+Umschalt+V) und drücken Sie die
   Eingabetaste.
3. Falls Git fehlt, zeigt das Installationsprogramm einen Befehl dafür an. Führen Sie ihn aus und
   wiederholen Sie Schritt 2.
4. Beantworten Sie die Fragen mit `y`. Das Installationsprogramm startet `desman` im Terminal.
   **DESMAN** erscheint im Anwendungsmenü und auf dem Desktop. Erlauben Sie den Start der
   Verknüpfung, wenn Sie danach gefragt werden.

Weitere Einzelheiten stehen in [install.md](install.md).

## 2. Erster Start und Zugriffsschlüssel

Die Ersteinrichtung läuft im Terminal. Bei der letzten Frage können Sie die experimentelle
Browseroberfläche wählen. Mit `desman interface terminal` oder `desman interface gui` ändern Sie
die Auswahl später.

DESMAN nutzt Modelle im Internet. Dafür brauchen Sie einen kostenlosen Schlüssel, eine Art
Passwort für das Programm. Beim ersten Start öffnet sich automatisch die Einrichtung.

1. Wählen Sie unter **Sprache** „Deutsch“ oder Ihre Sprache.
2. **Schlüssel.** Am einfachsten ist der Einstieg mit einem dieser Dienste. Beide sind kostenlos
   und Sie brauchen nur ein Konto:
   - **Groq** — https://console.groq.com/keys (anmelden, „Create API Key“ wählen, Schlüssel
     kopieren).
   - **Google AI Studio** — https://aistudio.google.com/apikey (mit dem Google-Konto anmelden,
     „Create API key“ wählen, Schlüssel kopieren).

   Fügen Sie den Schlüssel in das vorgesehene Feld ein. Dass die Zeichen verborgen sind, ist
   normal. Sie können Schlüssel mehrerer Anbieter hinzufügen. Ist das Limit eines Schlüssels
   erreicht, wechselt DESMAN zum nächsten.
   **Fügen Sie Schlüssel niemals in eine Chatnachricht ein**. Nutzen Sie nur das Schlüsselfeld.
3. **Berechtigungen.** Anfangs sind alle aus. Für den Test können Sie „das Netzwerk über die
   Modellanbieter hinaus zu nutzen“ aktivieren, damit Agenten öffentliche Webseiten lesen können.
   Lassen Sie die anderen aus, sofern der Betreiber nichts anderes sagt.
4. **desman-global.** Sie können „Jetzt nicht“ wählen und später beitreten (Abschnitt 5).

Die kostenlosen Limits sind klein und können sich ändern. Meldet DESMAN ein erreichtes Limit, geht
die Arbeit nach der Erneuerung weiter, meist nach einer Minute, manchmal nach einem Tag. Wenn
länger als eine Stunde keine Antwort kommt, melden Sie es (Abschnitt 7).

## 3. Bedienung

Die experimentelle Browseroberfläche ist so aufgebaut:

- **Links** stehen die Chats und unten die Schaltflächen **Einstellungen**, **Problem melden** und
  **Nachrichten**.
- **In der Mitte** ist der Chat. Schreiben Sie einfach, was Sie brauchen.
- **Rechts** stehen **Aufgaben** (woran DESMAN gerade arbeitet) und **Module** (welche Modelle
  arbeiten).

Einfache Fragen beantwortet DESMAN sofort. Aus einer größeren Bitte („Schreiben Sie …“, „Erstellen
Sie …“, „Finden Sie heraus …“) wird eine *Aufgabe*. DESMAN plant, erledigt und prüft die Schritte
und schildert das Ergebnis im Chat. Die Dateien liegen im Ordner der Aufgabe. DESMAN zeigt Ihnen
den Ort.

Scheitert ein Schritt, fragt DESMAN nach. Unter dem Chat erscheinen
**Mit dieser Antwort erneut versuchen**, **So akzeptieren** und **Als fehlgeschlagen markieren**.
Sie können auch schreiben, was korrigiert werden soll.

Änderungen an DESMAN selbst (Einstellungen, Berechtigungen, Schlüssel) werden erst vorgenommen,
wenn Sie in einem eigenen Fenster „Ja“ anklicken. Erscheint das Fenster unerwartet, klicken Sie
auf „Nein“.

Geben Sie im Chat `/` ein, um die Befehle zu sehen. `/help` öffnet die Hilfe, `/new` startet einen
neuen Chat. Sie können DESMAN auch einfach fragen, wie etwas geht.

## 4. Was Sie ausprobieren können

Nutzen Sie das Programm, wie Sie möchten. Ein paar Ideen:

1. Fragen Sie etwas Einfaches wie „Was können Sie?“ oder „Wie richte ich Sie ein?“
2. Bitten Sie um einen Text: „Schreiben Sie meinen Nachbarn einen Brief über das Treffen am
   Samstag“ oder „Planen Sie eine zweitägige Reise nach Dresden“.
3. Bitten Sie um Code: „Schreiben Sie ein Python-Programm, das die Wörter in einem Text zählt“.
   Unter Linux führt DESMAN es aus, wenn „Shell-Befehle auszuführen“ erlaubt ist.
4. Setzen Sie eine Aufgabe fort: „Fügen Sie jetzt … hinzu“.
5. Schließen Sie das Fenster mitten in einer Aufgabe und öffnen Sie es über die Verknüpfung
   wieder. Die Aufgabe sollte weiterlaufen.
6. Bitten Sie um eine Änderung, zum Beispiel „Antworten Sie kurz“. Ein Bestätigungsfenster sollte
   erscheinen.
7. Treten Sie dem Netzwerk bei (Abschnitt 5) und schreiben Sie einer anderen Person aus dem Test
   (Abschnitt 6).

Merken oder notieren Sie sich alles, was seltsam, unklar oder langsam war. Auch das sind Probleme.

## 5. Das Netzwerk desman-global

Mit `desman global donate searxng --calls N` verleihen Sie die verwaltete SearXNG-Suche. Ohne eigene Websuche schaltet `desman global search on|off` das Ausleihen ein oder aus.

desman-global verbindet die Rechner der Teilnehmenden. Ihr Rechner kann mit Ihren *verliehenen*
Modellen kleine Teile fremder Aufgaben erledigen. Für große Aufgaben fasst das Netzwerk freie
Rechner zu Gruppen zusammen. Ihre Aufgaben haben Vorrang. Fremde Arbeit läuft nur, wenn Ihr DESMAN
nichts zu tun hat.

**Beitreten:**

1. **Einstellungen** → **desman-global**.
2. Fügen Sie in **Registrierung** die Serveradresse ein, die Ihnen der Betreiber schickt. Sie
   beginnt mit `https://`.
3. Klicken Sie auf **Beitreten**. Der Fingerabdruck Ihres Knotens erscheint.

**Ein Modell verleihen** (freiwillig): Wählen Sie im selben Abschnitt ein Modell, setzen Sie
**Aufrufe pro Tag** und **Tokens pro Tag** und klicken Sie auf **Verleihen**. Mit **Zurücknehmen**
holen Sie es zurück. Zurzeit können Sie nur ein *lokales* Modell verleihen (Ollama auf Ihrem
Rechner). Die Bedingungen kostenloser Cloud-Schlüssel erlauben keine Arbeit für andere Personen.
Schlüssel bleiben auf Ihrem Rechner. Ihr DESMAN macht den Aufruf selbst und gibt nur das Ergebnis
weiter.

**Austreten:** Klicken Sie im selben Abschnitt auf **Verlassen**. Das Netzwerk erhält dann sofort
nichts mehr von Ihrem Rechner.

Das Netzwerk funktioniert, solange die DESMAN-Seite offen ist. Ist der Server nicht erreichbar,
arbeitet DESMAN allein weiter und versucht es später erneut.

## 6. Nachrichten

Teilnehmende können einander schreiben. Nachrichten sind verschlüsselt. Der Betreiber sieht, wer
wem wann schreibt, kann den Text aber nicht lesen.

1. **Name.** Geben Sie `/global name dein_name` im Chat ein. Erlaubt sind 3–32 Zeichen:
   lateinische Kleinbuchstaben, Ziffern, `_` und `-`. Andere sehen diesen Namen.
2. **Kontakt.** Sie können nur Personen schreiben, die ebenfalls zugestimmt haben. Senden Sie
   `/global contact add name`. Die andere Person nimmt mit demselben Befehl und Ihrem Namen an.
3. **Schreiben:** `/global send name "die Nachricht"`. Hat der Text mehrere Wörter, nutzen Sie
   gerade Anführungszeichen `"`. Das gilt auch für die folgenden Befehle.
4. **Lesen:** Klicken Sie links auf **Nachrichten** oder nutzen Sie `/messages` oder
   `/messages name`.
5. **Blockieren:** `/global block name`. Die andere Person wird nicht benachrichtigt.
   `/global unblock name` hebt die Sperre auf.
6. **Beschwerde:** `/global complain name "was passiert ist"`. Um dem Betreiber eine bestimmte
   Nachricht zu zeigen, ergänzen Sie `--message NUMMER`. Die Nummer wird beim Lesen angezeigt.
   **Der Text dieser Nachricht geht an den Betreiber**.

DESMAN führt Nachrichten anderer Personen nie als Aufgaben aus. Links und Dateien darin starten
nichts. Geht die Schlüsseldatei des Knotens verloren, etwa bei einer vollständigen Deinstallation,
gehen die Nachrichten verloren.

## 7. Ein Problem melden

Sie brauchen eine Verbindung zum Netzwerk (Abschnitt 5).

1. Klicken Sie unten links auf **Problem melden**.
2. Beschreiben Sie, was Sie getan und erwartet haben und was passiert ist. Zum Beispiel: „Ich habe
   um einen Brief gebeten und die Aufgabe hängt seit 20 Minuten bei Schritt 2 von 3“.
3. Im Fenster steht **genau der Text**, den der Betreiber erhält, mit Ihrer Beschreibung und den
   Versionen von Programm und System. Lesen Sie ihn und klicken Sie auf „Ja“.

Schlüssel gelangen nie in eine Meldung. Findet DESMAN einen Ihrer Schlüssel im Text, wird nichts
abgeschickt. Chats, Profil und Aufgabendateien werden ebenfalls nicht verschickt. Mit
`/global report "Beschreibung" --task NUMMER` fügen Sie den Verlauf einer bestimmten Aufgabe an.
Das Fenster zeigt genau, was hinzukommt.

Wenn Sie wegen eines Netzwerk- oder Startproblems nichts senden können, schreiben Sie dem
Betreiber auf dem üblichen Weg und fügen Sie nach Möglichkeit einen Screenshot bei.

## 8. Updates

Updates werden nur mit Ihrer Zustimmung installiert. Meldet der Betreiber eine neue Version,
verbinden Sie sich mit dem Netzwerk (Abschnitt 5). `/global update --show` zeigt die Version an,
`/global update --apply NUMMER` installiert sie. DESMAN zeigt die Änderungen und fragt nach.
Schließen Sie DESMAN danach und öffnen Sie es erneut. Weist das Programm auf „geschützte Dateien“
hin, geben Sie `/protected approve` ein und bestätigen Sie, um die neuen Regeln anzunehmen. Mit
`/global update --revert` kehren Sie zur vorigen Version zurück.

## 9. Deinstallation

- **Windows:** Schließen Sie DESMAN, drücken Sie Win+R, fügen Sie
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` ein und drücken Sie die Eingabetaste.
- **Linux:** Schließen Sie DESMAN und führen Sie
  `sh ~/.local/share/desman-app/install/install.sh --uninstall` in einem Terminal aus.

Das Programm wird entfernt. Einstellungen, Schlüssel und Verlauf bleiben für eine mögliche
Neuinstallation erhalten. Wenn Sie alles löschen wollen, ergänzen Sie unter Windows `-Purge` oder
unter Linux `--purge`. DESMAN zeigt, was genau entfernt wird, und fragt noch einmal nach.

Die angelegten Schlüssel können Sie auf den Websites der Anbieter widerrufen (Links in Abschnitt
2).
