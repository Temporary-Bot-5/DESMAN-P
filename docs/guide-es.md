# DESMAN(P): guía para participantes en la prueba

Gracias por ayudarnos. Aquí explicamos cómo instalar DESMAN, qué probar y cómo informar de un
problema. Puedes empezar por las secciones 1–3 y consultar el resto cuando lo necesites.

## 0. Lee esto primero

- **Es un prototipo.** Puede cerrarse de golpe, quedarse bloqueado, dar respuestas extrañas y
  perder trabajo sin terminar. No le confíes nada importante ni le encargues tareas que no puedas
  repetir. Los problemas que encuentres son el resultado principal de la prueba. Informa de todos
  ellos (sección 6).
- **Tus claves se quedan contigo.** La clave de acceso a un modelo (clave de API) se guarda solo
  en un archivo de tu equipo y solo tú puedes verla. No se envía a la red DESMAN, al propietario,
  a ningún modelo ni al historial del chat. A la red solo llega la *huella* de la clave, una
  cadena corta que permite distinguirla de otras sin poder recuperarla.
- **Solo tú activas la red.** Hasta que pulses **Unirse** (sección 5), DESMAN no se comunica con
  el servidor del propietario. Tus tareas, chats y archivos siempre se quedan en tu equipo. Las
  solicitudes solo van a los proveedores de los modelos cuyas claves añadiste.
- **Qué ve el operador (propietario del servidor) si te unes a la red:**
  - tu nodo: la huella de su clave, cuándo se conectó por última vez, qué modelos prestaste y con
    qué cuota diaria (proveedor, modelo y huella de la clave, nunca la clave);
  - el trabajo que hizo tu equipo para la red y su resultado;
  - los informes de problemas que enviaste, con el texto exacto de la ventana de confirmación
    (sección 6);
  - como cualquier sitio web, el servidor ve la dirección IP desde la que se accede.
- **Ejecutar código en Windows.** Instala e inicia Docker Desktop en modo de contenedores Linux
  y ejecuta `desman sandbox setup --yes` para autorizar la descarga de la imagen. Puedes habilitar
  el permiso para ejecutar código (`shell_execution`) en la configuración de DESMAN. Sin un
  entorno aislado, DESMAN puede escribir código y te pedirá ayuda para ejecutarlo.
  Para instalar Docker Desktop, usa la sección Docker del asistente de configuración o ejecuta `desman docker install --yes`.
  Sin Docker, `desman sandbox setup --yes` configura la zona aislada WSL en Windows si WSL2 está disponible, o WebAssembly para Python y pytest en los demás casos.

## 1. Instalación

El propietario te enviará el enlace a la página con los botones de instalación. Necesitas
internet, pero no permisos de administrador.

### Windows

1. Pulsa **Windows** en la página. Se descargará `install.bat`. Si el navegador pregunta si
   quieres quedarte con el archivo, elige «Conservar».
2. Abre Descargas y haz doble clic en `install.bat`. Si aparece la ventana azul «Windows protegió
   su PC», pulsa «Más información» → «Ejecutar de todas formas».
3. Se abrirá una ventana negra. El instalador te pedirá permiso para instalar lo necesario (uv,
   Python y Git si falta). Escribe `y` y pulsa Intro. Tardará unos minutos.
4. El instalador inicia `desman` en el terminal. El acceso directo **DESMAN** del escritorio
   también lo abre en un terminal.

### Linux

1. Pulsa **Linux** en la página y copia el comando que aparece.
2. Abre un terminal, pega el comando (Ctrl+Mayús+V) y pulsa Intro.
3. Si falta Git, el instalador mostrará el comando para instalarlo. Ejecútalo y repite el paso 2.
4. Responde `y` a las preguntas. El instalador inicia `desman` en el terminal. **DESMAN**
   aparece en el menú de aplicaciones y en el escritorio. Autoriza el acceso directo si te lo
   pide.

Encontrarás más detalles en [install.md](install.md).

## 2. Primer inicio y clave de acceso

La configuración inicial se realiza en el terminal. En la última pregunta puedes elegir la
interfaz experimental en el navegador. Puedes cambiar tu elección con `desman interface terminal` o `desman interface gui`.

DESMAN usa modelos que funcionan por internet. Para acceder necesitas una clave gratuita, algo
parecido a una contraseña para el programa. La primera vez se abre la configuración
automáticamente.

1. En **Idioma** elige «Español» o tu idioma.
2. **Clave.** Lo más sencillo es empezar con uno de estos servicios. Ambos son gratuitos y solo
   necesitas una cuenta:
   - **Groq** — https://console.groq.com/keys (inicia sesión, elige «Create API Key» y copia la
     clave).
   - **Google AI Studio** — https://aistudio.google.com/apikey (entra con tu cuenta de Google,
     elige «Create API key» y copia la clave).

   Pega la clave en el campo correspondiente de la configuración. Es normal que no se vean los
   caracteres. Puedes añadir claves de varios proveedores. Cuando alcanzas el límite de una,
   DESMAN pasa a otra. **Nunca pegues una clave en el chat**. Usa solo el campo para claves.
3. **Permisos.** Al principio todos están desactivados. Para la prueba puedes activar
   «usar la red más allá de los proveedores de modelos» para que los agentes lean páginas
   públicas. Deja el resto desactivado salvo que el propietario te indique otra cosa.
4. **desman-global.** Puedes elegir «ahora no» y unirte más adelante (sección 5).

Los límites gratuitos son bajos y pueden cambiar. Si DESMAN dice que alcanzaste un límite, el
trabajo seguirá cuando se renueve, normalmente al cabo de un minuto y a veces de un día. Si pasa
más de una hora sin respuesta, avisa (sección 6).

## 3. Cómo usarlo

En la interfaz experimental del navegador, la página se organiza así:

- **A la izquierda**, los chats. Abajo están **Ajustes**, **Informar de un problema** y
  **Mensajería**.
- **En el centro**, el chat. Escribe con tus propias palabras lo que necesitas.
- **A la derecha**, **Tareas** (qué está haciendo DESMAN) y **Módulos** (qué modelos están
  trabajando).

DESMAN responde enseguida a una pregunta sencilla. Una petición más amplia («escribe…»,
«prepara…», «averigua…») se convierte en una *tarea*. DESMAN planifica los pasos, los ejecuta, los
comprueba y te cuenta el resultado en el chat. Los archivos quedan en una carpeta propia de la
tarea y DESMAN te dice dónde está.

Si falla un paso, DESMAN te pregunta qué hacer. Debajo del chat aparecen
**Reintentar con esta respuesta**, **Aceptar tal cual** y **Marcar como fallida**. También puedes
explicar con tus palabras qué debe corregir.

Los cambios en DESMAN (ajustes, permisos y claves) solo se hacen cuando pulsas «Sí» en una ventana
aparte. Si aparece sin que la esperes, pulsa «No».

Escribe `/` en el chat para ver los comandos. `/help` abre la ayuda y `/new` inicia un chat nuevo.
También puedes preguntar directamente a DESMAN cómo se hace algo.

## 4. Qué probar

Úsalo como más te convenga. Para empezar:

1. Haz una pregunta sencilla, como «¿qué sabes hacer?» o «¿cómo te configuro?».
2. Pide un texto, por ejemplo «escribe una carta a mis vecinos sobre la reunión del sábado» o
   «planea un viaje de dos días a Granada».
3. Pide código, por ejemplo «escribe un programa en Python que cuente las palabras de un texto».
   En Linux, si activas «ejecutar comandos de shell», DESMAN también lo ejecutará.
4. Continúa una tarea con «ahora añade…».
5. Cierra la ventana a mitad de una tarea y vuelve a abrirla con el acceso directo. La tarea
   debería continuar.
6. Pide un cambio con tus propias palabras, como «responde de forma breve». Debería aparecer una
   ventana de confirmación.
7. Únete a la red (sección 5) y escribe a otro participante (sección 6).

Anota o recuerda todo lo que te parezca raro, confuso o lento. Eso también cuenta como problema.

## 5. La red desman-global

Usa `desman global donate searxng --calls N` para prestar búsquedas de SearXNG gestionado y `desman global search on|off` para habilitar o deshabilitar el préstamo cuando no tienes un buscador web.

desman-global conecta los equipos de los participantes. Tu equipo puede hacer pequeñas partes de
tareas ajenas con los modelos que hayas *prestado*. Para las tareas grandes, la red reúne los
equipos libres en grupos. Tus tareas tienen prioridad y las ajenas solo se ejecutan cuando tu
DESMAN está libre.

**Para unirte:**

1. **Ajustes** → **desman-global**.
2. En **Registro** pega la dirección del servidor que te dará el propietario. Empieza por
   `https://`.
3. Pulsa **Unirse**. Aparecerá la huella de tu nodo.

**Para prestar un modelo** (opcional), elige un modelo en ese mismo apartado, fija
**Llamadas al día** y **Tokens al día** y pulsa **Prestar**. Puedes retirarlo con **Recuperar**.
Por ahora solo se pueden prestar modelos *locales* (Ollama, en tu equipo). Las condiciones de las
claves gratuitas en la nube prohíben usarlas para trabajar para otras personas. Tus claves siempre
se quedan en tu equipo. Tu DESMAN hace la llamada y entrega a la red solo el resultado.

**Para salir**, pulsa **Salir** en ese mismo apartado. La red dejará de recibir datos de tu equipo
al instante.

La red funciona mientras la página de DESMAN está abierta. Si no puede acceder al servidor, DESMAN
sigue funcionando por su cuenta y vuelve a intentarlo más tarde.

## 6. Cómo informar de un problema

Si describes un problema con DESMAN en el chat, te ofrecerá un informe para que lo revises.

Para enviar un informe debes estar conectado a la red (sección 5).

1. Pulsa **Informar de un problema** abajo a la izquierda.
2. Explica qué hiciste, qué esperabas y qué pasó. Por ejemplo, «pedí una carta y la tarea lleva 20
   minutos atascada en el paso 2 de 3».
3. Verás el **texto exacto** que recibirá el operador, con tu descripción y las versiones del
   programa y el sistema. Léelo y pulsa «Sí».

Las claves no se incluyen en los informes. Si DESMAN detecta una clave tuya en el texto, cancela
el envío. Tampoco se envían chats, datos del perfil ni archivos. Para adjuntar el historial de una
tarea concreta, escribe `/global report --task NÚMERO "descripción"`. La ventana muestra
exactamente qué se añadirá.

Si no puedes enviarlo porque la red no conecta o el programa no abre, escribe al propietario por
el medio habitual y adjunta una captura de pantalla si puedes.

## 7. Actualizaciones

Las actualizaciones se instalan con tu permiso. Cuando el propietario te avise de una nueva
versión, conéctate a la red (sección 5). Usa `/global update --show` para verla y
`/global update --apply NÚMERO` para instalarla. DESMAN te muestra los cambios y te pide
confirmación. Después de instalarla, cierra y vuelve a abrir DESMAN. Si aparece un aviso sobre
«archivos protegidos», escribe `/protected approve` y confirma para aceptar la nueva versión de
las reglas. Para volver a la versión anterior, usa `/global update --revert`.

## 8. Desinstalación

- **Windows:** cierra DESMAN, pulsa Win+R, pega
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` y pulsa Intro.
- **Linux:** cierra DESMAN y ejecuta `sh ~/.local/share/desman-app/install/install.sh --uninstall`
  en un terminal.

En un terminal, `desman uninstall` muestra el plan de la misma desinstalación y `desman uninstall --yes` la confirma. También puedes usar Ajustes → General → Desinstalar DESMAN.

Se elimina el programa y se conservan los ajustes, las claves y el historial por si lo instalas de
nuevo. Para borrarlo todo, incluidas claves e historial, añade `-Purge` en Windows o `--purge` en
Linux. DESMAN te muestra qué borrará y vuelve a pedirte confirmación.

Puedes revocar las claves que creaste en los sitios de los proveedores (mismos enlaces que en la
sección 2).
