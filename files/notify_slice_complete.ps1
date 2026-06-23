# background 분할 완료 시 Windows 토스트 알림
param(
    [string]$TerminalFile = "C:\Users\give5\.cursor\projects\c-clipAI\terminals\588797.txt",
    [int]$PollSec = 30
)

function Send-Toast {
    param([string]$Title, [string]$Message)
    try {
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
        $template = @"
<toast duration="long">
  <visual>
    <binding template="ToastText02">
      <text id="1">$Title</text>
      <text id="2">$Message</text>
    </binding>
  </visual>
  <audio src="ms-winsoundevent:Notification.Default"/>
</toast>
"@
        $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $xml.LoadXml($template)
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("clipAI").Show($toast)
    }
    catch {
        [System.Reflection.Assembly]::LoadWithPartialName("System.Windows.Forms") | Out-Null
        [System.Windows.Forms.MessageBox]::Show($Message, $Title) | Out-Null
    }
}

Write-Host "[notify] watching: $TerminalFile"
while ($true) {
    if (Test-Path $TerminalFile) {
        $tail = Get-Content $TerminalFile -Tail 20 -ErrorAction SilentlyContinue
        $text = $tail -join "`n"
        if ($text -match "ended_at:") {
            if ($text -match "exit_code:\s*0") {
                Send-Toast "clipAI" "background 분할 완료! scan_clip_folders.py 실행하세요."
            }
            else {
                Send-Toast "clipAI" "background 분할 종료 (오류 가능). 터미널 로그를 확인하세요."
            }
            Write-Host "[notify] done"
            break
        }
    }
    Start-Sleep -Seconds $PollSec
}
