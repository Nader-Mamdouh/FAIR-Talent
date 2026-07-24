namespace FAIR.Application.DTOs.Chat
{
    public class MessageRequest
    {
        public string? Id { get; set; }
        public string? Content { get; set; }
        public string? SenderId { get; set; }
        public string? ReceiverId { get; set; }
        public bool IsRead { get; set; }
        public DateTime CreateDate { get; set; }
    }
}
